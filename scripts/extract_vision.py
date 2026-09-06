#!/usr/bin/env python3
"""Explicit TorchVision feature recipe; original checkpoints were not supplied.

This optional script is separate from the validated NumPy numerical runtime.
The selected weight enum determines preprocessing and is recorded verbatim.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder',choices=['resnet18','vit_b_16','vgg16'],required=True)
    p.add_argument('--weights',required=True,help='Explicit weight enum, e.g. IMAGENET1K_V1; DEFAULT is rejected')
    p.add_argument('--checkpoint',type=Path,help='Optional local state_dict; preprocessing still follows --weights')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New directory for features.npy and metadata.json')
    p.add_argument('--split',choices=['train','test','all'],default='all')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--device',default='cpu')
    p.add_argument('--download',action='store_true',help='Explicitly allow CIFAR-10 download')
    args=p.parse_args()
    if args.batch_size<1 or args.weights=='DEFAULT': p.error('positive batch size and explicit versioned weights required')
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import ConcatDataset, DataLoader
    from torchvision.models import get_model, get_model_weights
    from dcrl.data import sha256_file
    if args.output.exists() and any(args.output.iterdir()): p.error('output must be empty')
    weight=get_model_weights(args.encoder)[args.weights]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model=get_model(args.encoder,weights=None if args.checkpoint else weight)
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
    if args.encoder=='resnet18': model.fc=nn.Identity(); dim=512
    elif args.encoder=='vit_b_16': model.heads=nn.Identity(); dim=768
    else: model.classifier=nn.Sequential(*list(model.classifier.children())[:-1]); dim=4096
    model=model.to(args.device).eval()
    datasets=[]
    for split in (['train','test'] if args.split=='all' else [args.split]):
        datasets.append(torchvision.datasets.CIFAR10(root=args.data_root,train=split=='train',
                        transform=weight.transforms(),download=args.download))
    data=ConcatDataset(datasets)
    loader=DataLoader(data,batch_size=args.batch_size,shuffle=False,num_workers=0)
    args.output.mkdir(parents=True,exist_ok=True)
    features=np.lib.format.open_memmap(args.output/'features.npy',mode='w+',dtype='float32',shape=(len(data),dim))
    labels=np.empty(len(data),dtype=np.int64)
    offset=0
    with torch.inference_mode():
        for x,y in loader:
            z=model(x.to(args.device)).float().cpu().numpy()
            if z.shape!=(len(x),dim) or not np.isfinite(z).all(): raise ValueError('unexpected or nonfinite model output')
            features[offset:offset+len(x)]=z; labels[offset:offset+len(x)]=y.numpy(); offset+=len(x)
    features.flush(); del features
    np.save(args.output/'labels.npy',labels,allow_pickle=False)
    metadata={'recipe':'optional explicit feature reconstruction; not the original feature archive',
        'encoder':args.encoder,'weight_enum':args.weights,'checkpoint_sha256':sha256_file(args.checkpoint) if args.checkpoint else None,
        'preprocessing':str(weight.transforms()),'dataset':'CIFAR-10','split':args.split,
        'row_order':'train then test, original within-split order' if args.split=='all' else 'original split order',
        'shape':[len(data),dim],'torch':torch.__version__,'torchvision':torchvision.__version__,
        'feature_sha256':sha256_file(args.output/'features.npy'),'device_type':torch.device(args.device).type,
        'scope':'feature-spectrum diagnostics; combined splits must not be used as a supervised test protocol'}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(args.output/'features.npy')


if __name__=='__main__': main()
