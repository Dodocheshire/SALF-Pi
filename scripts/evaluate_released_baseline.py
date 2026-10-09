"""Evaluate the released 1000-class SALF checkpoint; no adaptation or selection."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from salf_pi.model import SpatialSALF
from data_utils import get_resnet_imagenet_preprocess
ROOT=Path(__file__).resolve().parents[1]

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=64);args=p.parse_args();torch.set_num_threads(8)
 directory=ROOT/'artifacts/released_baseline_v1';directory.mkdir(parents=True,exist_ok=True)
 root=Path('/home/pengcheng/VLG-CBM/datasets/imagenet/ILSVRC/Data/CLS-LOC');source=ImageFolder(root/'ImageNet_val',get_resnet_imagenet_preprocess())
 classes=json.loads((root/'ImageNet_train_200_10_clean/imagenet_class_index.json').read_text())
 mapping={entry[0]:int(index) for index,entry in classes.items()}
 assert len(source)==50000 and len(source.classes)==1000
 assert all(mapping[name]==index for name,index in source.class_to_idx.items()),'ImageFolder labels differ from official 1000-class indices'
 protocol={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'classes_sha256':hashlib.sha256(json.dumps(source.classes).encode()).hexdigest(),
           'checkpoint_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'artifacts/pretrained/imagenet').iterdir() if f.is_file()},
           'preprocess':'official SALF Resize(short side 224), CenterCrop224, ImageNet normalization',
           'batch_size':args.batch_size,'n':len(source),'training':False,'selection':False,'classification':'all 1000 released classes, not a 200-class remapping'}
 (directory/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 lock=ROOT.parent/'D-Pi/artifacts/resources/gpu_work.lock'
 with lock.open('a+') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX);model=SpatialSALF(ROOT/'artifacts/pretrained/imagenet').cuda().eval();correct=0;top5=0;nll=0.;offset=0;ps=[];ys=[];confidence=[]
  for images,labels in DataLoader(source,batch_size=args.batch_size,num_workers=8,pin_memory=True):
   logits,concepts,maps=model(images.cuda());probs=logits.softmax(-1).cpu();correct+=int(probs.argmax(1).eq(labels).sum());top5+=int(probs.topk(5,dim=1).indices.eq(labels[:,None]).any(1).sum());nll+=float(-probs[torch.arange(len(labels)),labels].clamp_min(1e-12).log().sum());offset+=len(labels)
   ps.append(probs);ys.append(labels)
   if offset%4096==0:print(offset,correct/offset,flush=True)
  probabilities=torch.cat(ps);labels=torch.cat(ys)
  torch.save({'probabilities':probabilities,'labels':labels,'indices':torch.arange(len(labels))},directory/'predictions.pt')
  summary={'correct':correct,'n':offset,'accuracy':correct/offset,'top5':top5/offset,'nll':nll/offset,
           'nec_gt_1e5':float((model.official.final.weight.abs()>1e-5).sum())/1000,'concepts':4741,'protocol':protocol,
           'limits':'Released pretrained checkpoint reproduction with upstream preprocessing; no new noise training and no concept/background ground-truth audit.'}
  (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');out=ROOT/'reports/released_baseline.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
