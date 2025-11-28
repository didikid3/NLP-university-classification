#!/usr/bin/env python3
"""eval_model.py

Simple evaluation runner that loads a model checkpoint (from the training
script) or an HF model name, prepares the validation DataLoader and runs
`evaluate()` from `train_classifier.py`.

Example:
  python eval_model.py --val splits/val.zst --mapping college_to_id.json \
    --model bert-base-uncased --checkpoint model_out/pytorch_model.pt --batch-size 16

The script prints metrics and can optionally save them to a JSON file or
log to Weights & Biases (if configured).
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
import wandb

# import helpers from training script
from train_classifier import (
    ZstdJsonDataset,
    collate_batch,
    load_mapping,
    BertForCollege,
    evaluate,
    autocast_context,
)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate saved classifier on a validation file')
    p.add_argument('--val', required=True, help='Validation .zst file')
    p.add_argument('--mapping', required=True, help='college_to_id.json mapping file')
    p.add_argument('--dist', required=False, help='Distribution JSON to build mapping if mapping missing')
    p.add_argument('--model', default='bert-base-uncased', help='Hugging Face model name or path (used for tokenizer and model init)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (.pt) to load model weights from')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--max-val', type=int, default=None, help='Limit number of val samples')
    p.add_argument('--device', type=str, default=None, help='Device to run on, e.g. cuda or cpu (default auto)')
    p.add_argument('--use-bf16', action='store_true', help='Use bfloat16 autocast during evaluation when supported')
    p.add_argument('--out-json', type=str, default=None, help='Optional path to save metrics JSON')
    p.add_argument('--wandb-project', type=str, default=None, help='Optional wandb project to log metrics to')
    p.add_argument('--no-wandb', action='store_true', help='Disable wandb logging even if installed')
    return p.parse_args()


def load_checkpoint_into_model(model, ckpt_path, device):
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f'Checkpoint {ckpt_path} not found')
    ckpt = torch.load(str(p), map_location='cpu')
    # support both full state dicts and wrapped dicts used by train
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(k.startswith('bert') or k in model.state_dict() for k in ckpt.keys()):
        # heuristically accept this as a raw state_dict
        state = ckpt
    else:
        # last resort: try to pull 'state_dict' or use as-is
        state = ckpt.get('state_dict', None) if isinstance(ckpt, dict) else None
        if state is None:
            # try loading whole object into model (may error)
            try:
                model.load_state_dict(ckpt)
                return model
            except Exception as e:
                raise RuntimeError(f'Unable to interpret checkpoint format: {e}')
    model.load_state_dict(state)
    return model


def main():
    args = parse_args()

    # device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load mapping
    mapping = load_mapping(args.mapping, args.dist, out_json=None)
    num_labels = len(mapping)

    # tokenizer + model init
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertForCollege(args.model, num_labels)

    # load checkpoint weights if provided
    if args.checkpoint:
        model = load_checkpoint_into_model(model, args.checkpoint, device)

    model.to(device)

    # prepare validation loader
    val_ds = ZstdJsonDataset(args.val, mapping, max_samples=args.max_val) if args.val else None
    if val_ds is None:
        raise SystemExit('Validation dataset not provided or not found')
    collate = lambda batch: collate_batch(batch, tokenizer, max_length=args.max_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)

    # optional wandb
    use_wandb = False
    if args.wandb_project and not args.no_wandb:
        try:
            import wandb
            use_wandb = True
            wandb.init(project=args.wandb_project)
        except Exception:
            use_wandb = False

    # run evaluation
    metrics = None
    if args.use_bf16 and device.type == 'cuda':
        with autocast_context(device, dtype=torch.bfloat16):
            metrics = evaluate(model, val_loader, device)
    else:
        metrics = evaluate(model, val_loader, device)

    print('Evaluation metrics:', metrics)

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f'Wrote metrics to {out_p}')

    if use_wandb:
        try:
            # send all metrics; evaluate() returns 'loss' so val/loss will be logged
            wandb.log({f'val/{k}': v for k, v in metrics.items() if v is not None})
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
