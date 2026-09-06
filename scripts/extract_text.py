#!/usr/bin/env python3
"""Extract local encoder features from explicit UTF-8 JSONL text records."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='JSONL records with a text string')
    p.add_argument('--model-dir',type=Path,required=True,help='Local pretrained model/tokenizer directory')
    p.add_argument('--pooling',choices=['cls','mean'],required=True)
    p.add_argument('--max-length',type=int,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if min(args.max_length,args.batch_size)<1: p.error('length and batch size must be positive')
    if args.output.exists() and any(args.output.iterdir()): p.error('output must be empty')
    if not args.model_dir.is_dir(): p.error('model-dir must be an existing local directory')
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from dcrl.data import sha256_file
    rows=[]
    with args.input.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row.get('text'),str): raise ValueError('every row must contain a text string')
            rows.append(row['text'])
    if not rows: p.error('input contains no text records')
    torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False).to(args.device).eval()
    args.output.mkdir(parents=True,exist_ok=True)
    dim=model.config.hidden_size
    features=np.lib.format.open_memmap(args.output/'features.npy',mode='w+',dtype='float32',shape=(len(rows),dim))
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            batch=tokenizer(rows[start:start+args.batch_size],return_tensors='pt',padding=True,truncation=True,max_length=args.max_length).to(args.device)
            hidden=model(**batch).last_hidden_state
            if args.pooling=='cls': z=hidden[:,0]
            else:
                mask=batch['attention_mask'].unsqueeze(-1)
                z=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            z=z.float().cpu().numpy()
            if not np.isfinite(z).all(): raise ValueError('nonfinite feature output')
            features[start:start+len(z)]=z
    features.flush(); del features
    files={str(p.relative_to(args.model_dir)):sha256_file(p) for p in sorted(args.model_dir.rglob('*')) if p.is_file() and p.suffix in {'.json','.txt','.model','.safetensors','.bin'}}
    meta={'recipe':'optional local feature reconstruction; original model revision and pooling were not supplied',
          'input_sha256':sha256_file(args.input),'model_files_sha256':files,'pooling':args.pooling,
          'max_length':args.max_length,'row_order':'input JSONL order','shape':[len(rows),dim],
          'feature_sha256':sha256_file(args.output/'features.npy'),'torch':torch.__version__,
          'transformers':transformers.__version__,'device_type':torch.device(args.device).type}
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(args.output/'features.npy')


if __name__=='__main__': main()
