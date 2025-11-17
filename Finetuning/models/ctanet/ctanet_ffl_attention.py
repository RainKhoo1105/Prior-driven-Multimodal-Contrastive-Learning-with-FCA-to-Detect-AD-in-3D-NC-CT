import sys

sys.path.append('../..')

import collections
import torch
import torch.nn as nn
from backbone import resnet

class CrossModalAttention3D(nn.Module):
    def __init__(self, in_channels, mag_channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        
        
        self.q_proj = nn.Conv3d(in_channels, in_channels, 1)
        self.k_proj = nn.Conv3d(in_channels, in_channels, 1)
        self.v_proj = nn.Conv3d(in_channels, in_channels, 1)
        
        
        self.mag_k_proj = nn.Conv3d(mag_channels, in_channels, 1)
        self.mag_v_proj = nn.Conv3d(mag_channels, in_channels, 1)
        
        
        self.out_proj = nn.Conv3d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x, mag):
        
        B, C = x.shape[:2]
        q = self.q_proj(x)  # [B,C,X,Y,Z]
        k = self.k_proj(x)  # [B,C,X,Y,Z]
        v = self.v_proj(x)  # [B,C,X,Y,Z]
        mag = nn.functional.interpolate(mag, size=x.shape[2:], mode='trilinear')
        mag_k = self.mag_k_proj(mag)  # [B,C,X,Y,Z]
        mag_v = self.mag_v_proj(mag)  # [B,C,X,Y,Z]
        
        q = q.view(B, self.num_heads, C//self.num_heads, *x.shape[2:])
        k = k.view(B, self.num_heads, C//self.num_heads, *x.shape[2:])
        v = v.view(B, self.num_heads, C//self.num_heads, *x.shape[2:])
        mag_k = mag_k.view(B, self.num_heads, C//self.num_heads, *x.shape[2:])
        mag_v = mag_v.view(B, self.num_heads, C//self.num_heads, *x.shape[2:])
        
        attn = (q * (k+mag_k)).sum(dim=2, keepdim=True) * self.scale
        attn = attn.softmax(dim=1)
        attn = self.dropout(attn)
        
        out = (attn * (v + mag_v))
        out = out.reshape(B, C, *x.shape[2:]) 
        return self.out_proj(out) + x  
    
class CTANet(nn.Module):
    def __init__(self, in_width, in_height, in_depth, n_class, no_cuda=False, gpu_id=['1'], freeze=False, unfreeze_num=0,pretrain_path=None,tag='resnet18'):
        super().__init__()
        self.in_width = in_width
        self.in_height = in_height
        self.in_depth = in_depth
        self.no_cuda=no_cuda
        self.n_class = n_class
        self.freeze = freeze
        self.unfreeze_num = unfreeze_num

        
        if tag == 'resnet18':
            self.ct_features = self._init_resnet_features(tag, pretrain_path)
            self.num_ftrs = 512
            self.ct_attention = CrossModalAttention3D(
                in_channels=self.num_ftrs,
                mag_channels=1,
                num_heads=8
            )
            self.clf = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=self.num_ftrs, out_features=n_class, bias=True)
            )
            
        elif tag == 'resnet34':
            self.ct_features = self._init_resnet_features(tag, pretrain_path)
            self.num_ftrs = 512
            self.ct_attention = CrossModalAttention3D(
                in_channels=self.num_ftrs,
                mag_channels=1,
                num_heads=8
            )
            self.clf = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)), 
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=self.num_ftrs, out_features=n_class, bias=True)
            )
            
        elif tag == 'resnet50':
            self.ct_features = self._init_resnet_features(tag, pretrain_path)
            self.num_ftrs = 2048
            self.ct_attention = CrossModalAttention3D(
                in_channels=self.num_ftrs,
                mag_channels=1,
                num_heads=8
            )
            self.clf = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)), 
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.num_ftrs, out_features=n_class, bias=True)
            )
        
        
        if pretrain_path is not None:
            self._load_pretrained_weights(pretrain_path)
        
        
        if freeze:
            self._freeze_layers(unfreeze_last_n=unfreeze_num)


    def _init_resnet_features(self, tag, pretrain_path):
        
        if tag == 'resnet18':
            resnet_model = resnet.resnet18(
                sample_input_W=self.in_width,
                sample_input_H=self.in_height,
                sample_input_D=self.in_depth,
                shortcut_type='B',
                no_cuda=self.no_cuda,
                num_seg_classes=self.n_class
            )
        elif tag == 'resnet34':
            resnet_model = resnet.resnet34(
                sample_input_W=self.in_width,
                sample_input_H=self.in_height,
                sample_input_D=self.in_depth,
                shortcut_type='B',
                no_cuda=self.no_cuda,
                num_seg_classes=self.n_class
            )
        elif tag == 'resnet50':
            resnet_model = resnet.resnet50(
                sample_input_W=self.in_width,
                sample_input_H=self.in_height,
                sample_input_D=self.in_depth,
                shortcut_type='B',
                no_cuda=self.no_cuda,
                num_seg_classes=self.n_class
            )
        
        
        return nn.Sequential(*list(resnet_model.children())[:-1])
    
        
    def _load_pretrained_weights(self, pretrain_path):
        
        print(f'loading pretrained weights: {pretrain_path}')
        pretrain = torch.load(pretrain_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        
        if isinstance(pretrain, collections.OrderedDict):
            state_dict = pretrain
        elif isinstance(pretrain, dict) and 'state_dict' in pretrain:
            state_dict = pretrain['state_dict']
        else:
            raise ValueError(f"no state_dict found in pretrained weights: {type(pretrain)}")
        
        
        ct_features_dict = {}
        ct_attention_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('ct_features.'):
                new_key = k.replace('ct_features.', '')
                ct_features_dict[new_key] = v
            elif k.startswith('ct_attention.'):
                new_key = k.replace('ct_attention.', '')
                ct_attention_dict[new_key] = v
        
        
        features_net_dict = self.ct_features.state_dict()
        features_pretrain_dict = {k: v for k, v in ct_features_dict.items() if k in features_net_dict.keys()}
        features_net_dict.update(features_pretrain_dict)
        self.ct_features.load_state_dict(features_net_dict, strict=False)
        
       
        attention_net_dict = self.ct_attention.state_dict()
        attention_pretrain_dict = {k: v for k, v in ct_attention_dict.items() if k in attention_net_dict.keys()}
        attention_net_dict.update(attention_pretrain_dict)
        self.ct_attention.load_state_dict(attention_net_dict, strict=False)
        
        print('pretrained weights loaded successfully')

    def _freeze_layers(self, unfreeze_last_n=0):
        
        
        for param in self.ct_attention.parameters():
            param.requires_grad = False
        
        for param in self.ct_features.parameters():
            param.requires_grad = False
        
        
        if unfreeze_last_n > 0:
            children = list(self.ct_features.children())
            num_children = len(children)
            
            
            for child in children[max(0, num_children - unfreeze_last_n):]:
                for param in child.parameters():
                    param.requires_grad = True
        
        for param in self.clf.parameters():
            param.requires_grad = True
        
        print(f'freeze ct_features except last {unfreeze_last_n} layers')    


    
    def tofft(self,ct):
        
        fft_ct =  torch.fft.rfftn(ct, dim=(-3, -2,-1))
        fft_ct_mag = torch.abs(fft_ct)
        fft_ct_phase = torch.angle(fft_ct)

       
        fft_ct_mag = (fft_ct_mag - torch.min(fft_ct_mag)) / (torch.max(fft_ct_mag) - torch.min(fft_ct_mag))
        fft_ct_phase = (fft_ct_phase + torch.pi) % (2 * torch.pi) - torch.pi 

        return fft_ct_mag,fft_ct_phase
    
    def forward(self, x):
        ct_features = self.ct_features(x)
        
        mag, _ = self.tofft(x)
        
        attention_output = self.ct_attention(ct_features, mag)
        
        out = self.clf(attention_output)
        
        return out
