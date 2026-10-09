"""Keep upstream SALF parameters/pooling; expose the spatial injection point."""
import torch
from torch import nn
from cbm import load_cbm


class SpatialSALF(nn.Module):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        self.official=load_cbm(str(checkpoint),'resnet50_imagenet',device)
        self.official.eval().requires_grad_(False)
        # Upstream stats are plain tensors. Register copies for portable device moves.
        self.register_buffer('concept_mean',self.official.proj_mean.reshape(1,-1))
        self.register_buffer('concept_std',self.official.proj_std.reshape(1,-1))

    def train(self,mode=True):
        super().train(mode);self.official.eval();return self

    def encode(self,images):
        return self.official.backbone(images).last_hidden_state

    def classify_spatial(self,features):
        if features.shape[-2:]!=self.official.map_size:features=self.official.resize_layer(features)
        maps=self.official.proj_layer(features)
        # Avoid upstream squeeze() dropping the batch axis for N=1.
        concepts=(self.official.pool_layer(maps).flatten(1)-self.concept_mean)/self.concept_std
        return self.official.final(concepts),concepts,maps

    def forward(self,images):return self.classify_spatial(self.encode(images))


class SpatialGaussian(nn.Module):
    """Local + global context predicts a mean and std before SALF spatial pooling.

    Fresh initialization, never a trained-performance claim. No labels, masks,
    or evaluation statistics enter the generator.
    """
    def __init__(self,channels=2048,hidden=64,initial_scale=.02):
        super().__init__()
        self.encoder=nn.Conv2d(channels,hidden,1)
        self.context=nn.Linear(channels,hidden,bias=False)
        self.mean_head=nn.Conv2d(hidden,channels,1)
        self.scale_head=nn.Conv2d(hidden,channels,1)
        nn.init.zeros_(self.mean_head.weight);nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias,float(torch.log(torch.expm1(torch.tensor(initial_scale)))))
        self.register_buffer('feature_scale',torch.ones(1,channels,1,1))

    def distribution(self,features):
        normalized=features/self.feature_scale
        global_context=self.context(normalized.mean((2,3)))[:,:,None,None]
        h=torch.nn.functional.gelu(self.encoder(normalized)+global_context)
        return self.mean_head(h)*self.feature_scale,torch.nn.functional.softplus(self.scale_head(h))*self.feature_scale

    def forward(self,features,mode='stochastic',generator=None):
        if mode=='clean':return features,features.new_zeros(())
        mean,scale=self.distribution(features)
        if mode=='mean':scale=torch.zeros_like(scale)
        elif mode=='sigma_only':mean=torch.zeros_like(mean)
        elif mode!='stochastic':raise ValueError(mode)
        eps=torch.randn(features.shape,device=features.device,dtype=features.dtype,generator=generator)
        energy=((mean/self.feature_scale).square()+(scale/self.feature_scale).square()).mean()
        return features+mean+scale*eps,energy
