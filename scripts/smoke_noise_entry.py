"""Real released checkpoint parity and gradient smoke; not a trained experiment."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from salf_pi.model import SpatialSALF,SpatialGaussian
from data_utils import get_resnet_imagenet_preprocess

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/pretrained/imagenet');args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(0)
    path=ROOT/'artifacts/resources/gpu_work.lock'
    # Share the LF/D lock with other experiments, regardless of this module's path.
    lock=ROOT.parent/'D-Pi/artifacts/resources/gpu_work.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        model=SpatialSALF(args.checkpoint).cuda().eval()
        data=ImageFolder('/home/pengcheng/VLG-CBM/datasets/imagenet/ILSVRC/Data/CLS-LOC/ImageNet_val',get_resnet_imagenet_preprocess())
        # Fixed examples by original index, not prediction success. No use for training.
        ids=[0,50,500,5000];images=torch.stack([data[i][0] for i in ids]).cuda()
        with torch.no_grad():
            official=model.official(images);wrapped=model(images)
            errors=[float((a-b).abs().max()) for a,b in zip(official,wrapped)]
            assert max(errors)<1e-5,errors
            one=model(images[:1]);assert one[0].shape==(1,1000)
            features=model.encode(images)
        generator=SpatialGaussian().cuda();generator.feature_scale.copy_(features.square().mean((0,2,3),keepdim=True).sqrt().clamp_min(.01))
        # This scale is for the smoke test only. The scientific protocol requires
        # calibration from training data and forbids using this test scale.
        altered,energy=generator(features,mode='stochastic',generator=torch.Generator(device='cuda').manual_seed(0))
        logits,concepts,maps=model.classify_spatial(altered)
        target=model(images)[0].detach().argmax(1)
        loss=torch.nn.functional.cross_entropy(logits,target)+.001*energy;loss.backward()
        gradient={name:float(p.grad.norm()) for name,p in generator.named_parameters() if p.grad is not None}
        assert gradient['mean_head.weight']>0 and gradient['scale_head.weight']>0
        assert all(p.grad is None for p in model.parameters())
        torch.save({'images':images.cpu(),'clean_maps':wrapped[2].cpu(),'random_maps':maps.detach().cpu(),'concepts':(args.checkpoint/'concepts.txt').read_text().splitlines()},ROOT/'artifacts/smoke_maps.pt')
        result={'status':'PASS','upstream_commit':'e45f8645afde37a8a7ae66db6987b5a8a53552a0','checkpoint_sha256':{p.name:sha(p) for p in args.checkpoint.iterdir() if p.is_file()},
                'official_parity_max_abs':errors,'batch1_shape':list(one[0].shape),'spatial_shape':list(features.shape),'concept_map_shape':list(maps.shape),
                'generator_grad_norms':gradient,'fixed_test_indices':ids,'trained_noise':False,'accuracy_evaluated':False,
                'limits':'Four-image implementation/gradient smoke only. Smoke feature scale uses these images; it must not be reused for scientific training/evaluation.',
                'source_sha256':sha(__file__),'torch_version':torch.__version__}
        out=ROOT/'reports/entry_smoke.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
