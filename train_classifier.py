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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import zstandard as zstd

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


class ZstdJsonDataset(IterableDataset):
    """Streams newline JSON objects from a .zst file and yields dicts."""

    def __init__(self, path, mapping, max_samples=None, text_key_priority=('title','selftext')):
        self.path = Path(path)
        self.mapping = mapping
        self.max_samples = max_samples
        self.text_key_priority = text_key_priority

    def __iter__(self):
        dctx = zstd.ZstdDecompressor()
        with open(self.path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                it = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                for i, line in enumerate(it, start=1):
                    if self.max_samples is not None and i > self.max_samples:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    sub = obj.get('subreddit') or obj.get('subreddit_name')
                    if sub is None:
                        continue
                    label = self.mapping.get(sub)
                    if label is None:
                        # skip unknown labels
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
                    yield {'text': text, 'label': label}


def collate_batch(batch, tokenizer, max_length=256):
    texts = [b['text'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    enc['labels'] = labels
    return enc


class BertForCollege(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

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


def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = correct / total if total else 0.0
    try:
        from sklearn.metrics import f1_score
        micro = f1_score(trues, preds, average='micro')
        macro = f1_score(trues, preds, average='macro')
    except Exception:
        micro = macro = None
    return {'accuracy': acc, 'micro_f1': micro, 'macro_f1': macro}


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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapping = load_mapping(args.mapping, args.dist, out_json=args.mapping if args.mapping and not Path(args.mapping).exists() else None)
    num_labels = len(mapping)
    print(f'Number of labels: {num_labels}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertForCollege(args.model, num_labels)
    model.to(device)

    train_ds = ZstdJsonDataset(args.train, mapping, max_samples=args.max_train)
    val_ds = ZstdJsonDataset(args.val, mapping, max_samples=args.max_val) if args.val else None

    collate = lambda batch: collate_batch(batch, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate) if val_ds else None
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
    total_steps = args.epochs * (args.max_train // args.batch_size if args.max_train else 1000)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(1, total_steps))

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        # compute total steps for tqdm if we could count available records
        train_total_steps = None
        if available_train is not None:
            effective = available_train if args.max_train is None else min(available_train, args.max_train)
            train_total_steps = math.ceil(effective / args.batch_size) if effective is not None else None
        for step, batch in enumerate(tqdm(train_loader, desc=f'train epoch {epoch}', total=train_total_steps)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step and step % args.log_steps == 0:
                print(f'Epoch {epoch} step {step} loss {running_loss / args.log_steps:.4f}')
                running_loss = 0.0

        if val_loader is not None:
            # provide tqdm total for evaluation as well
            val_total_steps = None
            if available_val is not None:
                effective_val = available_val if args.max_val is None else min(available_val, args.max_val)
                val_total_steps = math.ceil(effective_val / args.batch_size) if effective_val is not None else None
            # build an iterator that passes total into tqdm inside evaluate()
            metrics = evaluate(model, val_loader, device)
            print(f'Validation metrics after epoch {epoch}:', metrics)
            score = metrics['micro_f1'] if metrics['micro_f1'] is not None else metrics['accuracy']
            if score is not None and score > best_val:
                best_val = score
                out_dir = Path(args.outdir)
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f'Saving model to {out_dir}')
                # save state dict only
                torch.save({'model_state_dict': model.state_dict(), 'mapping': mapping}, out_dir / 'pytorch_model.pt')


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
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max-train', type=int, default=None, help='Limit number of train samples for quick runs')
    p.add_argument('--max-val', type=int, default=None, help='Limit number of val samples')
    p.add_argument('--log-steps', type=int, default=100)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
