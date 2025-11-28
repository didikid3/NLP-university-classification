#!/usr/bin/env python3
"""
train_classifier.py

Train a BERT-based classifier (ModernBERT or other Hugging Face model) with
a single linear classification head (logistic regression layer) on top of the
pooled output. Data is expected as newline-delimited JSON inside Zstandard
compressed files (same format used earlier: records contain 'title', 'selftext', 'subreddit', 'id').

This script streams data (memory-efficient) using an IterableDataset.

Example:
  python train_classifier.py --train splits/train.zst --val splits/val.zst \
    --mapping college_to_id.json --model bert-base-uncased --outdir model_out --epochs 1 --batch-size 16

Notes:
- `--model` can be 'ModernBERT' if available on HF; otherwise use 'bert-base-uncased'.
- Make sure `college_to_id.json` exists (created earlier). If not, provide `--dist` to build mapping.
"""

import argparse
import io
import json
import math
import os
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, IterableDataset

import zstandard as zstd

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from contextlib import nullcontext

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    from typing import Any
    def tqdm(it: Any, **kwargs: Any) -> Any:
        return it


import wandb


class ZstdJsonDataset(IterableDataset):
    """Streams newline JSON objects from a .zst file and yields dicts.

    Supports skipping a number of matching records (useful for resuming mid-epoch).
    """

    def __init__(self, path, mapping, max_samples=None, text_key_priority=('title','selftext'), skip: int = 0):
        self.path = Path(path)
        self.mapping = mapping
        self.max_samples = max_samples
        self.text_key_priority = text_key_priority
        # number of matching records to skip (only counts records that would be yielded)
        self.skip = int(skip or 0)

    def __iter__(self):
        dctx = zstd.ZstdDecompressor()
        # First, attempt to read as a zstd-compressed stream. If that fails we
        # fall back to reading the file as plain text (useful if file is not
        # compressed or contains a bad/corrupt zstd frame).
        try:
            with open(self.path, 'rb') as fh:
                with dctx.stream_reader(fh) as reader:
                    it = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                    matched = 0
                    for raw_line in it:
                        try:
                            obj = json.loads(raw_line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        sub = obj.get('subreddit') or obj.get('subreddit_name')
                        if sub is None:
                            continue
                        label = self.mapping.get(sub)
                        if label is None:
                            continue
                        # build text: prefer title + separator + selftext
                        parts = []
                        for k in self.text_key_priority:
                            v = obj.get(k)
                            if v:
                                parts.append(v)
                        text = '\n'.join(parts).strip()
                        if not text:
                            continue
                        # this is a matching record we would yield
                        matched += 1
                        if self.max_samples is not None and matched > self.max_samples:
                            break
                        # skip initial matching records when resuming
                        if matched <= self.skip:
                            continue
                        yield {'text': text, 'label': label}
            return
        except zstd.ZstdError as e:
            print(f'Warning: zstd decompression failed for {self.path}: {e}; attempting plain-text fallback')

        # Plain-text fallback (open as text file). Use a separate open call so
        # we don't try to seek/reuse a previously-closed binary file handle.
        with open(self.path, 'r', encoding='utf-8', errors='replace') as fh_text:
            matched = 0
            for raw_line in fh_text:
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                sub = obj.get('subreddit') or obj.get('subreddit_name')
                if sub is None:
                    continue
                label = self.mapping.get(sub)
                if label is None:
                    continue
                parts = []
                for k in self.text_key_priority:
                    v = obj.get(k)
                    if v:
                        parts.append(v)
                text = '\n'.join(parts).strip()
                if not text:
                    continue
                matched += 1
                if self.max_samples is not None and matched > self.max_samples:
                    break
                if matched <= self.skip:
                    continue
                yield {'text': text, 'label': label}


def collate_batch(batch, tokenizer, max_length=256):
    texts = [b['text'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    enc['labels'] = labels
    return enc


class ListDataset(torch.utils.data.Dataset):
    """Wrap a list of examples (dicts) so it can be used with DataLoader."""
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def autocast_context(device, dtype=torch.bfloat16):
    """Return an autocast context manager compatible with the installed PyTorch.

    Prefers the unified `torch.autocast` API when available (newer PyTorch),
    otherwise falls back to `torch.cuda.amp.autocast` for CUDA devices. If
    autocast isn't available or the device isn't CUDA, returns a no-op context.
    """
    if device is None or device.type != 'cuda':
        return nullcontext()
    # try unified API first
    try:
        autocast_fn = getattr(torch, 'autocast')
        return autocast_fn(device_type='cuda', dtype=dtype)
    except Exception:
        pass
    # fallback to cuda amp autocast
    try:
        return torch.cuda.amp.autocast(dtype=dtype)
    except Exception:
        return nullcontext()


class BertForCollege(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1, class_weight=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        # register class weight as a buffer so it moves with the model device
        if class_weight is not None:
            cw = torch.tensor(class_weight, dtype=torch.float)
            # ensure proper shape
            if cw.ndim == 0:
                cw = cw.unsqueeze(0)
            self.register_buffer('class_weight', cw)
        else:
            self.register_buffer('class_weight', None)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Some models (e.g. ModernBERT) do not accept `token_type_ids` as a kwarg.
        # Only pass token_type_ids if it's not None to avoid unexpected-kwarg errors.
        bert_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            bert_kwargs['token_type_ids'] = token_type_ids
        out = self.bert(**bert_kwargs)
        # use pooled output if available, otherwise take CLS token
        pooled = getattr(out, 'pooler_output', None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            # use class weights if provided
            cw = self.class_weight if isinstance(getattr(self, 'class_weight', None), torch.Tensor) else None
            if cw is not None:
                loss_fct = nn.CrossEntropyLoss(weight=cw)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits


def load_mapping(mapping_path=None, dist_path=None, out_json=None):
    if mapping_path and Path(mapping_path).exists():
        p = Path(mapping_path)
        # support Python module providing a mapping (e.g. college_to_id.py)
        if p.suffix == '.py':
            import importlib.util
            spec = importlib.util.spec_from_file_location('college_mapping_module', str(p))
            if spec is None or spec.loader is None:
                raise SystemExit(f'Unable to load python module from {p}')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # try common attribute names
            for attr in ('COLLEGE_TO_ID', 'mapping', 'COLLEGE_TO_ID_MAP'):
                if hasattr(mod, attr):
                    return getattr(mod, attr)
            raise SystemExit(f'No mapping dict found in python module {p}; expected COLLEGE_TO_ID or mapping')
        else:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            # support previous format {"mapping": {...}}
            if isinstance(obj, dict) and 'mapping' in obj:
                return obj['mapping']
            return obj
    if dist_path and Path(dist_path).exists():
        with open(dist_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        dist = obj.get('distribution', obj)
        # build mapping sorted by probability desc
        items = sorted(dist.items(), key=lambda kv: kv[1].get('probability', 0) if isinstance(kv[1], dict) else float(kv[1]), reverse=True)
        mapping = {k: i for i, (k, _) in enumerate(items)}
        if out_json:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump({'mapping': mapping}, f, ensure_ascii=False, indent=2)
        return mapping
    raise FileNotFoundError('No mapping or distribution file found')


def evaluate(model, dataloader, device, max_samples=None):
    """Evaluate `model` on `dataloader`.

    If `max_samples` is provided and > 0, evaluation will stop after approximately
    `max_samples` examples (the final batch will be sliced to exactly reach the
    requested count).
    """
    # preserve current training/eval mode so we can restore it after evaluation
    was_training = model.training
    model.eval()
    total = 0
    correct = 0
    preds = []
    trues = []
    total_loss_sum = 0.0
    max_samples = None if (max_samples is None or max_samples <= 0) else int(max_samples)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # evaluation forward; if caller wants bf16 autocast they will wrap around evaluate
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(logits, dim=-1)

            # If we have a max_samples limit, possibly slice the final batch so we
            # return metrics computed on exactly that many samples.
            if max_samples is not None:
                remaining = max_samples - total
                if remaining <= 0:
                    break
                if labels.size(0) > remaining:
                    logits = logits[:remaining]
                    pred = pred[:remaining]
                    labels = labels[:remaining]

            # compute summed loss for the (possibly sliced) batch so we can
            # report an average loss over all evaluated samples
            try:
                cw = getattr(model, 'class_weight', None)
                cw = cw if isinstance(cw, torch.Tensor) else None
                if cw is not None:
                    loss_fct = nn.CrossEntropyLoss(reduction='sum', weight=cw)
                else:
                    loss_fct = nn.CrossEntropyLoss(reduction='sum')
                batch_loss_sum = loss_fct(logits, labels).item()
            except Exception:
                batch_loss_sum = 0.0

            total_loss_sum += float(batch_loss_sum)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            batch_n = labels.size(0)
            total += batch_n
            correct += (pred == labels).sum().item()
            if max_samples is not None and total >= max_samples:
                break
    acc = correct / total if total else 0.0
    try:
        from sklearn.metrics import f1_score
        micro = f1_score(trues, preds, average='micro')
        macro = f1_score(trues, preds, average='macro')
    except Exception:
        micro = macro = None
    avg_loss = (total_loss_sum / total) if total else 0.0
    # restore previous mode
    try:
        if was_training:
            model.train()
        else:
            model.eval()
    except Exception:
        pass
    return {'loss': avg_loss, 'accuracy': acc, 'micro_f1': micro, 'macro_f1': macro}


def count_records(path, mapping=None, max_limit=None):
    """Count records in a .zst newline-JSON file that have a label in mapping.

    - If `mapping` is provided, only lines whose subreddit exists in mapping are counted.
    - If `max_limit` is provided, counting stops early when the limit is reached.
    """
    p = Path(path)
    if not p.exists():
        return 0
    dctx = zstd.ZstdDecompressor()
    c = 0
    with open(p, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            it = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            for line in it:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if mapping is not None:
                    sub = obj.get('subreddit') or obj.get('subreddit_name')
                    if sub is None or sub not in mapping:
                        continue
                c += 1
                if max_limit is not None and c >= max_limit:
                    break
    return c


def reservoir_sample(path, mapping, sample_size, text_key_priority=('title', 'selftext')):
    """Return a random sample (list of dicts) of up to `sample_size` matching records
    from a newline-JSON (possibly .zst) file using reservoir sampling. This is a
    single-pass, memory-efficient sampler suitable for large files.
    """
    p = Path(path)
    if not p.exists() or sample_size <= 0:
        return []
    sample_size = int(sample_size)
    dctx = zstd.ZstdDecompressor()
    reservoir = []
    seen = 0
    try:
        with open(p, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                it = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                for raw_line in it:
                    try:
                        obj = json.loads(raw_line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    sub = obj.get('subreddit') or obj.get('subreddit_name')
                    if sub is None:
                        continue
                    label = mapping.get(sub)
                    if label is None:
                        continue
                    parts = []
                    for k in text_key_priority:
                        v = obj.get(k)
                        if v:
                            parts.append(v)
                    text = '\n'.join(parts).strip()
                    if not text:
                        continue
                    seen += 1
                    entry = {'text': text, 'label': label}
                    if len(reservoir) < sample_size:
                        reservoir.append(entry)
                    else:
                        # replace with decreasing probability
                        import random as _rnd
                        j = _rnd.randrange(seen)
                        if j < sample_size:
                            reservoir[j] = entry
    except zstd.ZstdError:
        # fallback to plain text
        with open(p, 'r', encoding='utf-8', errors='replace') as fh_text:
            for raw_line in fh_text:
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                sub = obj.get('subreddit') or obj.get('subreddit_name')
                if sub is None:
                    continue
                label = mapping.get(sub)
                if label is None:
                    continue
                parts = []
                for k in text_key_priority:
                    v = obj.get(k)
                    if v:
                        parts.append(v)
                text = '\n'.join(parts).strip()
                if not text:
                    continue
                seen += 1
                entry = {'text': text, 'label': label}
                if len(reservoir) < sample_size:
                    reservoir.append(entry)
                else:
                    import random as _rnd
                    j = _rnd.randrange(seen)
                    if j < sample_size:
                        reservoir[j] = entry
    return reservoir


def compute_label_counts(path, mapping, num_labels, max_limit=None):
    """Count label occurrences in a newline-JSON (.zst) file.

    Returns a list of length `num_labels` with integer counts for each label id.
    """
    p = Path(path)
    counts = [0] * int(num_labels)
    if not p.exists():
        return counts
    dctx = zstd.ZstdDecompressor()
    seen = 0
    try:
        with open(p, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                it = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                for raw_line in it:
                    try:
                        obj = json.loads(raw_line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    sub = obj.get('subreddit') or obj.get('subreddit_name')
                    if sub is None:
                        continue
                    label = mapping.get(sub)
                    if label is None:
                        continue
                    if 0 <= int(label) < num_labels:
                        counts[int(label)] += 1
                    seen += 1
                    if max_limit is not None and seen >= max_limit:
                        break
    except zstd.ZstdError:
        with open(p, 'r', encoding='utf-8', errors='replace') as fh_text:
            for raw_line in fh_text:
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                sub = obj.get('subreddit') or obj.get('subreddit_name')
                if sub is None:
                    continue
                label = mapping.get(sub)
                if label is None:
                    continue
                if 0 <= int(label) < num_labels:
                    counts[int(label)] += 1
                seen += 1
                if max_limit is not None and seen >= max_limit:
                    break
    return counts


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # reproducibility: seed various RNGs if requested
    if getattr(args, 'seed', None) is not None:
        seed = int(args.seed)
        # Python random
        random.seed(seed)
        # numpy
        try:
            np.random.seed(seed)
        except Exception:
            pass
        # torch
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        # set deterministic algorithms where possible
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            # PyTorch 1.8+ has this API
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        # set PYTHONHASHSEED for best-effort reproducibility (must be set before process start to be fully effective)
        try:
            os.environ['PYTHONHASHSEED'] = str(seed)
        except Exception:
            pass
    # Optionally enable TF32 / higher float32 matmul precision for speed on Ampere+ GPUs.
    if getattr(args, 'enable_tf32', False) and torch.cuda.is_available():
        try:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        except Exception:
            print("Some issue setting up TF 32")
    # bfloat16 usage: check if requested and supported
    use_bf16 = False
    if getattr(args, 'use_bf16', False) and torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, 'is_bf16_supported', None)
        try:
            supported = is_bf16_supported() if callable(is_bf16_supported) else False
        except Exception:
            supported = False
        if supported:
            use_bf16 = True
            print('bfloat16 autocast enabled for training/eval')
        else:
            print('Warning: --use-bf16 requested but device/PyTorch does not report bf16 support; continuing without bf16')
    mapping = load_mapping(args.mapping, args.dist, out_json=args.mapping if args.mapping and not Path(args.mapping).exists() else None)
    num_labels = len(mapping)
    print(f'Number of labels: {num_labels}')

    # handle class weight specification (None / 'balanced' / JSON path / CSV)
    class_weight_tensor = None
    if getattr(args, 'class_weight', None):
        cw_arg = args.class_weight
        # balanced: compute inverse-frequency weights from training data
        if cw_arg == 'balanced':
            print('Computing balanced class weights from training data (this may scan the train file)')
            counts = compute_label_counts(args.train, mapping, num_labels, max_limit=(args.max_train if args.max_train else None))
            total = float(sum(counts)) if sum(counts) > 0 else 0.0
            if total > 0:
                weights = [ (total / (num_labels * c)) if c > 0 else 0.0 for c in counts ]
            else:
                weights = [1.0] * num_labels
            class_weight_tensor = torch.tensor(weights, dtype=torch.float)
        else:
            # cw_arg might be a path to a JSON file, or comma-separated list of floats
            p = Path(cw_arg)
            if p.exists():
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        obj = json.load(f)
                    if isinstance(obj, list) and len(obj) == num_labels:
                        weights = [float(x) for x in obj]
                    elif isinstance(obj, dict):
                        # mapping from label-name or id to weight
                        weights = [0.0] * num_labels
                        for k, v in obj.items():
                            try:
                                idx = int(k)
                            except Exception:
                                idx = mapping.get(k)
                            if idx is not None and 0 <= int(idx) < num_labels:
                                weights[int(idx)] = float(v)
                    else:
                        raise ValueError('Unexpected JSON format for class weights')
                    class_weight_tensor = torch.tensor(weights, dtype=torch.float)
                except Exception as e:
                    raise SystemExit(f'Unable to read class-weight JSON from {p}: {e}')
            else:
                # try parse comma-separated floats
                try:
                    parts = [float(x) for x in cw_arg.split(',')]
                    if len(parts) != num_labels:
                        raise ValueError('Number of class-weight values does not match number of labels')
                    class_weight_tensor = torch.tensor(parts, dtype=torch.float)
                except Exception as e:
                    raise SystemExit(f'Unable to parse --class-weight argument: {e}')

    # initialize wandb if requested and available
    use_wandb = (args.wandb_project is not None) and (wandb is not None) and (not args.no_wandb)
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run)
        # log config
        wandb.config.update({k: v for k, v in vars(args).items() if not k.startswith('_')})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertForCollege(args.model, num_labels, class_weight=(class_weight_tensor.tolist() if class_weight_tensor is not None else None))
    model.to(device)

    # validation dataset (no skipping needed)
    val_ds = ZstdJsonDataset(args.val, mapping, max_samples=args.max_val) if args.val else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b, tokenizer, max_length=args.max_length)) if val_ds else None

    collate = lambda batch: collate_batch(batch, tokenizer, max_length=args.max_length)
    # Count available records (respecting mapping) to provide tqdm totals/ETAs.
    # This performs a quick pass over the compressed file(s); it's fast relative
    # to a full training run and gives accurate step counts for progress bars.
    try:
        available_train = count_records(args.train, mapping=mapping, max_limit=None) if args.train else 0
    except Exception:
        available_train = None
    try:
        available_val = count_records(args.val, mapping=mapping, max_limit=None) if args.val else 0
    except Exception:
        available_val = None

    if available_train is not None:
        print(f'Available train records (matching mapping): {available_train:,}')
    if available_val is not None:
        print(f'Available val records (matching mapping):   {available_val:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # rough total steps for scheduler (used if scheduler state not loaded)
    total_steps = args.epochs * (args.max_train // args.batch_size if args.max_train else 1000)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=max(1, total_steps))

    best_val = -1.0

    # global step counter used for WandB logging (monotonic across epochs)
    global_step = 0

    # compute effective number of train samples per epoch (if available)
    effective = None
    if available_train is not None:
        effective = available_train if args.max_train is None else min(available_train, args.max_train)

    # Resume support: load checkpoint if provided
    start_epoch = 1
    samples_processed = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f'Loading checkpoint {ckpt_path} to resume')
            ckpt = torch.load(str(ckpt_path), map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception:
                    print('Warning: failed to fully load optimizer state; continuing with fresh optimizer')
            if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception:
                    print('Warning: failed to fully load scheduler state')
            start_epoch = int(ckpt.get('epoch', 1))
            samples_processed = int(ckpt.get('samples_processed', 0))
            # compute skip inside epoch (number of matching records already consumed in this epoch)
            skip_in_epoch = 0
            if effective is not None and effective > 0:
                skip_in_epoch = samples_processed % effective
                # if the checkpoint was exactly at epoch boundary, start next epoch
                if skip_in_epoch == 0 and samples_processed > 0:
                    start_epoch = start_epoch + 1
            else:
                skip_in_epoch = 0
            print(f'Resuming from epoch {start_epoch} with {samples_processed} samples processed (skip_in_epoch={skip_in_epoch})')
        else:
            print(f'Warning: resume checkpoint {args.resume} not found; starting fresh')
            skip_in_epoch = 0
    else:
        skip_in_epoch = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        # prepare train dataset for this epoch; on the first resumed epoch we may skip some records
        this_epoch_skip = skip_in_epoch if epoch == start_epoch else 0
        train_ds = ZstdJsonDataset(args.train, mapping, max_samples=args.max_train, skip=this_epoch_skip)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate)

        # compute total steps for tqdm for this epoch if we could count available records
        train_total_steps = None
        if effective is not None:
            effective_for_epoch = effective - (this_epoch_skip if epoch == start_epoch else 0)
            train_total_steps = math.ceil(effective_for_epoch / args.batch_size) if effective_for_epoch is not None else None

        for step, batch in enumerate(tqdm(train_loader, desc=f'train epoch {epoch}', total=train_total_steps)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if use_bf16 and device.type == 'cuda':
                with autocast_context(device, dtype=torch.bfloat16):
                    loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # increment global_step once per optimization step/batch
            global_step += 1
            running_loss += loss.item()
            # update samples processed (actual number in this batch)
            batch_count = labels.size(0)
            samples_processed += int(batch_count)
            # periodic save
            if args.save_steps and samples_processed and samples_processed % args.save_steps == 0:
                out_dir = Path(args.outdir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = out_dir / f'checkpoint_samples_{samples_processed}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'mapping': mapping,
                    'epoch': epoch,
                    'samples_processed': samples_processed
                }, ckpt_path)
                print(f'Saved periodic checkpoint to {ckpt_path} (samples_processed={samples_processed})')
                if use_wandb:
                    wandb.save(str(ckpt_path))

            # periodic evaluation (mid-epoch)
            if args.eval_steps and samples_processed and samples_processed % args.eval_steps == 0 and val_loader is not None:
                # For mid-epoch eval we may want to use a random subset to avoid
                # repeatedly evaluating the same head-of-file examples (which can
                # bias metrics). If `--eval-random` is set and `--eval-samples` > 0,
                # draw a reservoir sample and evaluate on that.
                mid_metrics = None
                if getattr(args, 'eval_random', False) and args.eval_samples and args.eval_samples > 0:
                    sampled = reservoir_sample(args.val, mapping, args.eval_samples)
                    if len(sampled) == 0:
                        print('Warning: eval_random requested but no validation samples found; skipping mid-epoch eval')
                    else:
                        tmp_loader = DataLoader(ListDataset(sampled), batch_size=args.batch_size, collate_fn=collate)
                        if use_bf16 and device.type == 'cuda':
                            with autocast_context(device, dtype=torch.bfloat16):
                                mid_metrics = evaluate(model, tmp_loader, device)
                        else:
                            mid_metrics = evaluate(model, tmp_loader, device)
                else:
                    if use_bf16 and device.type == 'cuda':
                        with autocast_context(device, dtype=torch.bfloat16):
                            mid_metrics = evaluate(model, val_loader, device, max_samples=(args.eval_samples if args.eval_samples and args.eval_samples > 0 else None))
                    else:
                        mid_metrics = evaluate(model, val_loader, device, max_samples=(args.eval_samples if args.eval_samples and args.eval_samples > 0 else None))
                if mid_metrics is not None:
                    print(f'Validation metrics at samples {samples_processed}:', mid_metrics)
                    if use_wandb:
                        log_dict = {f'val/{k}': v for k, v in mid_metrics.items() if v is not None}
                        try:
                            wandb.log(log_dict, step=global_step)
                        except Exception:
                            wandb.log(log_dict)

            if step and step % args.log_steps == 0:
                avg_loss = running_loss / args.log_steps
                print(f'Epoch {epoch} step {step} loss {avg_loss:.4f}')
                # wandb log train loss (use global_step for a consistent x-axis)
                if use_wandb:
                    try:
                        wandb.log({'train/loss': avg_loss}, step=global_step)
                    except Exception:
                        wandb.log({'train/loss': avg_loss})
                running_loss = 0.0

        # evaluate at epoch end (if validation set provided)
        metrics = None
        if val_loader is not None:
            # provide tqdm total for evaluation as well
            val_total_steps = None
            if available_val is not None:
                effective_val = available_val if args.max_val is None else min(available_val, args.max_val)
                val_total_steps = math.ceil(effective_val / args.batch_size) if effective_val is not None else None
            if use_bf16 and device.type == 'cuda':
                with autocast_context(device, dtype=torch.bfloat16):
                    metrics = evaluate(model, val_loader, device)
            else:
                metrics = evaluate(model, val_loader, device)
            print(f'Validation metrics after epoch {epoch}:', metrics)
            if use_wandb:
                log_dict = {f'val/{k}': v for k, v in metrics.items() if v is not None}
                try:
                    wandb.log(log_dict, step=global_step)
                except Exception:
                    wandb.log(log_dict)

        # always save an epoch checkpoint (and also save best when improved)
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch_save_path = out_dir / f'pytorch_model_epoch{epoch}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mapping': mapping,
            'epoch': epoch,
            'samples_processed': samples_processed,
            'metrics': metrics
        }, epoch_save_path)
        print(f'Saved epoch checkpoint to {epoch_save_path}')
        if use_wandb:
            wandb.save(str(epoch_save_path))

        # save best checkpoint based on validation score (if available)
        if metrics is not None:
            score = metrics['micro_f1'] if metrics['micro_f1'] is not None else metrics['accuracy']
            if score is not None and score > best_val:
                best_val = score
                best_save_path = out_dir / 'pytorch_model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'mapping': mapping,
                    'epoch': epoch,
                    'samples_processed': samples_processed,
                    'metrics': metrics
                }, best_save_path)
                print(f'Saved best model to {best_save_path}')
                if use_wandb:
                    wandb.save(str(best_save_path))
        else:
            # no validation metrics available; optionally update best by other heuristics
            pass

    if use_wandb:
        wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description='Train BERT classifier for college subreddit classification')
    p.add_argument('--train', required=True, help='Train .zst file')
    p.add_argument('--val', required=False, help='Validation .zst file')
    p.add_argument('--mapping', required=False, help='college_to_id.json mapping file (if missing use --dist)')
    p.add_argument('--dist', required=False, help='college_probability_distribution.json to build mapping if mapping missing')
    p.add_argument('--model', default='bert-base-uncased', help='Hugging Face model name or path')
    p.add_argument('--outdir', default='model_out', help='Directory to save model checkpoints')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-train', type=int, default=None, help='Limit number of train samples for quick runs')
    p.add_argument('--max-val', type=int, default=None, help='Limit number of val samples')
    p.add_argument('--log-steps', type=int, default=100)
    p.add_argument('--save-steps', type=int, default=1000, help='Save a periodic checkpoint every N samples processed')
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    p.add_argument('--eval-steps', type=int, default=0, help='Run evaluation every N samples processed (0 to disable)')
    p.add_argument('--eval-samples', type=int, default=0, help='If >0, evaluate only this many validation samples (speeds up mid/epoch eval)')
    p.add_argument('--eval-random', action='store_true', help='Use random reservoir-sampled validation subset for mid-epoch evaluation')
    p.add_argument('--enable-tf32', action='store_true', help='Enable TF32 / float32 matmul precision for faster training on Ampere+ GPUs')
    p.add_argument('--use-bf16', action='store_true', help='Use bfloat16 autocast for training/eval when supported by device')
    p.add_argument('--wandb-project', type=str, default=None, help='W&B project name to log to')
    p.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (team/user)')
    p.add_argument('--wandb-run', type=str, default=None, help='W&B run name (optional)')
    p.add_argument('--no-wandb', action='store_true', help='Disable wandb even if installed')
    p.add_argument('--class-weight', type=str, default=None, help="Specify class weights: 'balanced' to compute inverse-frequency, path to JSON (list or dict), or comma-separated floats")
    p.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducibility (sets python/numpy/torch seeds)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
