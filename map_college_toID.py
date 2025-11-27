#!/usr/bin/env python3
"""map_college_toID.py

Read `college_probability_distribution.json` and produce a stable mapping from
college/subreddit name -> integer ID. Writes a JSON mapping and an optional
small Python module for easy import.

Usage:
  python college_classification/map_college_toID.py \
	  --dist ../college_probability_distribution.json \
	  --out-json ../college_to_id.json \
	  --out-py ../college_to_id.py
"""

import json
from pathlib import Path
import argparse


def build_mapping(dist_obj, order='prob_desc'):
	# dist_obj expected shape: { 'total_records': N, 'distribution': {col: {count, probability}} }
	if 'distribution' in dist_obj:
		dist_map = dist_obj['distribution']
	else:
		dist_map = dist_obj

	# create list of colleges in stable order
	if order == 'prob_desc':
		items = sorted(dist_map.items(), key=lambda kv: kv[1].get('probability', 0) if isinstance(kv[1], dict) else float(kv[1]), reverse=True)
	else:
		items = sorted(dist_map.items(), key=lambda kv: kv[0])

	mapping = {}
	for i, (col, info) in enumerate(items):
		mapping[col] = i
	return mapping


def parse_args():
	p = argparse.ArgumentParser(description='Create mapping from college (subreddit) to integer id')
	p.add_argument('--dist', required=True, help='Path to college_probability_distribution.json')
	p.add_argument('--out-json', default='college_to_id.json', help='Output JSON mapping file')
	p.add_argument('--out-py', default=None, help='Optional output Python module (writes a dict named COLLEGE_TO_ID)')
	p.add_argument('--order', choices=['prob_desc','alpha'], default='prob_desc', help='Ordering for assigning ids')
	return p.parse_args()


def main():
	args = parse_args()
	dist_path = Path(args.dist)
	if not dist_path.exists():
		raise SystemExit(f'Distribution file not found: {dist_path}')

	with dist_path.open('r', encoding='utf-8') as f:
		dist_obj = json.load(f)

	mapping = build_mapping(dist_obj, order=args.order)

	out_json = Path(args.out_json)
	with out_json.open('w', encoding='utf-8') as f:
		json.dump({'mapping': mapping}, f, ensure_ascii=False, indent=2)

	print(f'Wrote mapping for {len(mapping):,} colleges to: {out_json}')

	if args.out_py:
		out_py = Path(args.out_py)
		with out_py.open('w', encoding='utf-8') as f:
			f.write('# Auto-generated college -> id map\n')
			f.write('COLLEGE_TO_ID = ')
			json.dump(mapping, f, ensure_ascii=False, indent=2)
			f.write('\n')
		print(f'Wrote Python module to: {out_py}')


if __name__ == '__main__':
	main()

