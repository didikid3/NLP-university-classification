#!/usr/bin/env python3
import argparse
import io
import json
import importlib.util
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import zstandard as zstd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm
import wandb


def load_mapping_py(path):
    spec = importlib.util.spec_from_file_location("college_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COLLEGE_TO_ID

def cleanup_checkpoints(outdir: Path, max_to_keep: int):
    if max_to_keep <= 0:
        return  # keep everything

    # list step_* and epoch_* dirs only
    ckpts = []
    for d in outdir.iterdir():
        if d.is_dir() and (d.name.startswith("step_") or d.name.startswith("epoch_")):
            ckpts.append(d)

    # nothing to do
    if len(ckpts) <= max_to_keep:
        return

    # sort naturally by the number after step_ or epoch_
    def extract_num(path: Path):
        name = path.name
        num = name.split("_")[-1]
        try: return int(num)
        except: return 0

    ckpts_sorted = sorted(ckpts, key=extract_num)

    # remove older ones
    remove_count = len(ckpts_sorted) - max_to_keep
    to_delete = ckpts_sorted[:remove_count]

    for ckpt in to_delete:
        print(f"[Checkpoint Rotation] Removing old checkpoint: {ckpt}")
        try:
            # remove full directory
            import shutil
            shutil.rmtree(ckpt)
        except Exception as e:
            print(f"Warning: failed to delete {ckpt}: {e}")


# ---------------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------------

class ZstdJsonDataset(IterableDataset):
    def __init__(self, path, mapping, max_samples=None, skip: int = 0):
        self.path = Path(path)
        self.mapping = mapping
        self.max_samples = max_samples
        self.skip = int(skip or 0)

    def __iter__(self):
        dctx = zstd.ZstdDecompressor()
        matched = 0
        count = 0

        with open(self.path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                it = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

                for line in it:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    sub = obj.get("subreddit") or obj.get("subreddit_name")
                    if sub is None:
                        continue
                    label = self.mapping.get(sub)
                    if label is None:
                        continue

                    text = (obj.get("title") or "") + "\n" + (obj.get("selftext") or "")
                    text = text.strip()
                    if not text:
                        continue

                    # this is a matching record we would yield
                    matched += 1
                    if self.max_samples and matched > self.max_samples:
                        return

                    # skip initial matching examples when resuming
                    if matched <= self.skip:
                        continue

                    count += 1
                    yield {"text": text, "label": label}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate(batch, tokenizer, max_length):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = labels
    return enc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # bf16 / mixed precision setup
    # -------------------------------
    bf16_supported = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    use_bf16 = bf16_supported
    use_amp = (not bf16_supported) and torch.cuda.is_available()

    print(f"bf16 supported: {bf16_supported}")
    print(f"Using bf16: {use_bf16}")
    print(f"Using FP16 AMP: {use_amp}")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # -------------------------------
    # Load mapping
    # -------------------------------
    mapping = load_mapping_py(args.mapping)
    num_labels = len(mapping)

    # -------------------------------
    # Load tokenizer (from resume if available)
    # -------------------------------
    tokenizer_source = None
    if args.resume:
        resume_path = Path(args.resume)
        # If resume is a checkpoint directory that contains tokenizer files, prefer it
        if resume_path.is_dir():
            # HuggingFace tokenizer files are typically tokenizer.json or vocab files
            if (resume_path / "tokenizer.json").exists() or (resume_path / "vocab.txt").exists() or (resume_path / "tokenizer_config.json").exists():
                tokenizer_source = str(resume_path)
    if tokenizer_source is None:
        tokenizer_source = args.model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    # -------------------------------
    # Load model
    # -------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels
    ).to(device)

    # enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # REQUIRED for checkpointing
    print("Gradient checkpointing ENABLED")

    # -------------------------------
    # Resume support: training state
    # -------------------------------
    start_epoch = 1
    global_step = 0
    samples_processed = 0
    resume_skip = 0
    at_epoch_end = False

    if getattr(args, 'resume', None):
        resume_path = Path(args.resume)
        state_path = None
        # If resume is a directory containing training_state.pt, prefer that
        if resume_path.is_dir() and (resume_path / 'training_state.pt').exists():
            state_path = resume_path / 'training_state.pt'
            # also try to load model weights from the resume dir if present
            if (resume_path / 'pytorch_model.bin').exists() or (resume_path / 'config.json').exists():
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(str(resume_path)).to(device)
                    print(f"Loaded model weights from checkpoint dir: {resume_path}")
                except Exception:
                    print("Warning: failed to load model via from_pretrained(resume); will attempt state_dict load if provided.")
        elif resume_path.is_file():
            state_path = resume_path

        if state_path is not None and state_path.exists():
            print(f"Resuming training state from {state_path}")
            ckpt = torch.load(str(state_path), map_location=device)
            # restore model weights if present in state file
            try:
                if 'model_state_dict' in ckpt and ckpt['model_state_dict'] is not None:
                    model.load_state_dict(ckpt['model_state_dict'])
                    print("Loaded model_state_dict from training_state.pt")
            except Exception:
                print('Warning: failed to load model_state_dict from resume checkpoint')

            # restore optimizer/scheduler/scaler state if present (we will create optimizer/scheduler below)
            saved_optimizer_state = ckpt.get('optimizer_state_dict', None)
            saved_scheduler_state = ckpt.get('scheduler_state_dict', None)
            saved_scaler_state = ckpt.get('scaler_state_dict', None)

            # bookkeeping
            start_epoch = int(ckpt.get('epoch', 1))
            global_step = int(ckpt.get('global_step', 0))
            samples_processed = int(ckpt.get('samples_processed', 0)) if 'samples_processed' in ckpt else global_step * args.batch_size
            at_epoch_end = bool(ckpt.get('at_epoch_end', False))
            resume_skip = int(ckpt.get('skip', 0)) if 'skip' in ckpt else 0

            print(f"Resume: start_epoch={start_epoch}, global_step={global_step}, samples_processed={samples_processed}, skip={resume_skip}, at_epoch_end={at_epoch_end}")

            # restore scaler if provided
            if saved_scaler_state is not None and scaler is not None:
                try:
                    scaler.load_state_dict(saved_scaler_state)
                    print("Restored GradScaler state from checkpoint")
                except Exception:
                    print("Warning: failed to restore scaler state; continuing with fresh scaler")
        else:
            print(f'Warning: resume path {args.resume} not found or invalid; starting fresh')

    # -------------------------------
    # Data (build datasets/loaders AFTER resume handling so skip is honored)
    # -------------------------------
    train_ds = ZstdJsonDataset(args.train, mapping, max_samples=args.max_train, skip=resume_skip)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate(b, tokenizer, args.max_length)
    )

    val_loader = None
    if args.val:
        val_ds = ZstdJsonDataset(args.val, mapping, max_samples=args.max_val)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate(b, tokenizer, args.max_length)
        )

    # Create a small mid-epoch validation loader if mid-epoch eval is enabled
    mid_eval_loader = None
    if args.mid_eval_steps > 0 and val_loader is not None:
        mid_eval_batches = []
        for i, batch in enumerate(val_loader):
            mid_eval_batches.append(batch)
            if i + 1 >= args.mid_eval_batches:
                break
        mid_eval_loader = mid_eval_batches

    # -------------------------------
    # Optimizer + Scheduler (create AFTER model is finalized)
    # -------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = max(1, args.epochs * (1000 if args.max_train is None else max(1, args.max_train // args.batch_size)))
    warmup_steps = int(0.06 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # restore optimizer / scheduler states if available
    if 'saved_optimizer_state' in locals() and saved_optimizer_state is not None:
        try:
            optimizer.load_state_dict(saved_optimizer_state)
            print("Restored optimizer state from checkpoint")
        except Exception:
            print("Warning: failed to restore optimizer state; continuing with fresh optimizer")

    if 'saved_scheduler_state' in locals() and saved_scheduler_state is not None:
        try:
            scheduler.load_state_dict(saved_scheduler_state)
            print("Restored scheduler state from checkpoint")
        except Exception:
            print("Warning: failed to restore scheduler state; continuing with fresh scheduler")

    # -------------------------------
    # wandb
    # -------------------------------
    if args.wandb:
        wandb.init(project=args.wandb_project or "modernbert", config=vars(args))

    # -------------------------------
    # Periodic saving setup
    # -------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Save the tokenizer once at the top-level outdir for convenience / reproducibility
    tokenizer.save_pretrained(outdir)

    # -----------------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------------
    epoch_iter = range(start_epoch, args.epochs + 1)
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0

        # if we resumed and the checkpoint indicated we were at_epoch_end=True,
        # we should start from the next epoch (handled via start_epoch logic above)
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            # ---------------------
            # Mixed precision forward
            # ---------------------
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(use_bf16 or use_amp)):
                outputs = model(**batch)
                loss = outputs.loss

            # ---------------------
            # Backward
            # ---------------------
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            running_loss += loss.item()
            samples_processed += batch["labels"].size(0) if "labels" in batch else args.batch_size

            # ---------------------
            # Logging
            # ---------------------
            if step % args.log_steps == 0 and step > 0:
                avg_loss = running_loss / args.log_steps
                print(f"[Epoch {epoch} Step {step}] loss={avg_loss:.4f}")

                if args.wandb:
                    wandb.log({"train_loss": avg_loss, "step": global_step})

                running_loss = 0.0

            # ---------------------
            # Periodic Checkpoint Saving
            # ---------------------
            if args.save_steps and global_step % args.save_steps == 0:
                ckpt_dir = outdir / f"step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # save HF-format model + tokenizer (tokenizer already saved to outdir but we duplicate to ckpt dir)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

                # also save training state for resume (optimizer/scheduler/scaler)
                state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'epoch': epoch,
                    'global_step': global_step,
                    'samples_processed': samples_processed,
                    'at_epoch_end': False,
                    'skip': 0
                }
                torch.save(state, ckpt_dir / 'training_state.pt')

                print(f"[Checkpoint] Saved model+state to: {ckpt_dir}")
                cleanup_checkpoints(outdir, args.max_checkpoints)
            
            # ---------------------
            # Mid-epoch evaluation
            # ---------------------
            if args.mid_eval_steps and global_step % args.mid_eval_steps == 0:
                if mid_eval_loader:
                    model.eval()
                    m_correct = 0
                    m_total = 0
                    m_loss = 0.0

                    with torch.no_grad():
                        for batch in mid_eval_loader:
                            batch = {k: v.to(device) for k, v in batch.items()}

                            with torch.autocast(device_type="cuda",
                                                dtype=autocast_dtype,
                                                enabled=(use_bf16 or use_amp)):
                                outputs = model(**batch)

                            loss = outputs.loss
                            preds = outputs.logits.argmax(dim=-1)

                            m_loss += loss.item() * batch["labels"].size(0)
                            m_correct += (preds == batch["labels"]).sum().item()
                            m_total += batch["labels"].size(0)

                    m_avg_loss = m_loss / m_total if m_total else 0.0
                    m_acc = m_correct / m_total if m_total else 0.0

                    print(f"[Mid-Epoch Eval] step={global_step} | acc={m_acc:.4f} | loss={m_avg_loss:.4f}")

                    if args.wandb:
                        wandb.log({
                            "val_acc": m_acc,
                            "val_loss": m_avg_loss,
                            "step": global_step
                        })

                    model.train()

        # -------------------------------------------------------------------
        # EVAL after each epoch
        # -------------------------------------------------------------------
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation"):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(use_bf16 or use_amp)):
                        outputs = model(**batch)

                    loss = outputs.loss
                    preds = outputs.logits.argmax(dim=-1)

                    val_loss += loss.item() * batch["labels"].size(0)
                    correct += (preds == batch["labels"]).sum().item()
                    total += batch["labels"].size(0)

            avg_val_loss = val_loss / total if total else 0.0
            acc = correct / total if total else 0.0
            print(f"Validation accuracy: {acc:.4f} | loss={avg_val_loss:.4f}")

            if args.wandb:
                wandb.log({"val_acc": acc, "val_loss": avg_val_loss, "step": global_step})

        # -------------------------------------------------------------------
        # Save end-of-epoch checkpoint
        # -------------------------------------------------------------------
        epoch_dir = outdir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

        # save training state for resume
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'epoch': epoch,
            'global_step': global_step,
            'samples_processed': samples_processed,
            'at_epoch_end': True,
            'skip': 0
        }
        torch.save(state, epoch_dir / 'training_state.pt')

        print(f"Saved epoch checkpoint to: {epoch_dir}")
        cleanup_checkpoints(outdir, args.max_checkpoints)

    if args.wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=False)
    p.add_argument("--mapping", required=True)
    p.add_argument("--resume", type=str, default=None, help='Path to checkpoint file or checkpoint directory to resume from')

    p.add_argument("--model", default="answerdotai/ModernBERT-base")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)

    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)

    p.add_argument("--log-steps", type=int, default=1000)
    p.add_argument("--save-steps", type=int, default=200000,
                   help="Save model every N training steps")
    p.add_argument("--max-checkpoints", type=int, default=5,
               help="Keep only the most recent N checkpoints (0=keep all)")


    p.add_argument("--outdir", default="modernbert_out")

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
