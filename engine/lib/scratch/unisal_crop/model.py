"""Unmodified upstream network, strict verified checkpoint, video-state adapter."""
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
from PIL import Image

def preprocess(frame):
    import torch
    if frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8:
        raise ValueError('Expected uint8 BGR frame')
    image=Image.fromarray(frame[:,:,::-1].copy()).resize((384,224),Image.Resampling.LANCZOS)
    x=np.asarray(image,dtype=np.float32)/255.
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(x.transpose(2,0,1).copy())

class Unisal:
    def __init__(self,root):
        import torch
        root=Path(root).resolve()
        manifest_path=root/'ACQUISITION.json'
        if manifest_path.is_file():
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            for item in manifest['files']:
                if hashlib.sha256((root/item['path']).read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Upstream file hash mismatch '+item['path'])
        for req in ('unisal/model.py','training_runs/pretrained_unisal/UNISAL.json','training_runs/pretrained_unisal/weights_best.pth'):
            if not (root/req).is_file():
                raise FileNotFoundError(f'Missing required UNISAL file: {root/req}')
        pkg=types.ModuleType('unisal');pkg.__path__=[str(root/'unisal')]
        if 'unisal' in sys.modules and list(getattr(sys.modules['unisal'],'__path__',[]))!=pkg.__path__:
            raise ValueError('Unexpected unisal already imported')
        sys.modules.setdefault('unisal',pkg)
        from unisal.model import UNISAL
        cfg=json.loads((root/'training_runs/pretrained_unisal/UNISAL.json').read_text(encoding='utf-8'))
        cfg['cnn_cfg']['pretrained']=False;cfg['verbose']=0
        model=UNISAL(**cfg).eval()
        state=torch.load(root/'training_runs/pretrained_unisal/weights_best.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        self.parameters=sum(p.numel() for p in model.parameters())
        if self.parameters!=3714872:
            raise ValueError(f'Unexpected UNISAL parameter count: {self.parameters}')
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.torch=torch;self.model=model.to(self.device)

    def predict_batches(self,batches):
        hidden=None
        with self.torch.inference_mode():
            for batch in batches:
                ids,frames=zip(*batch)
                x=self.torch.stack(frames).unsqueeze(0).to(self.device)
                output,hidden=self.model(x,source='DHF1K',static=False,h0=hidden,return_hidden=True)
                maps=output[0,:,0].float().cpu().numpy()
                if len(maps)!=len(ids):raise ValueError('Model output length mismatch')
                for fid,m in zip(ids,maps):yield fid,m
