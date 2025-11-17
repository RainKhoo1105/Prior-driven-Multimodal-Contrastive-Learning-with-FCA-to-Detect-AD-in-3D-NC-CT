

import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
import torch
from models.resnet import resnet50
from models.resnet import resnet34
from models.resnet import resnet18
from models import resnet
from transformers import AutoModel
from .FANLayer import FANLayer

def add_dropout_to_resnet(resnet, dropout_rate=0.5):
    
    modules = list(resnet.children())[:-1]
    modules.append(nn.Dropout(dropout_rate))
    modules.append(list(resnet.children())[-1])
    new_resnet = nn.Sequential(*modules)
    return new_resnet
class CrossModalAttention3D(nn.Module):
    def __init__(self, in_channels, mag_channels, num_heads=8, dropout=0.1):
        super().__init__()
        # Initialize layers and parameters, coming soon
        

    def forward(self, x, mag):
        
        # Implement the forward pass, coming soon
    
        return self.out_proj(out) + x  
    
# Create the BertClassfier class
class ModelCLR(nn.Module):
    def __init__(
        self, in_width, in_height, in_depth, res_base_model, n_class,
        bert_base_model, out_dim, freeze_layers, do_lower_case, pretrain_path=None
    ):
        super(ModelCLR, self).__init__()
        self.in_width = in_width
        self.in_height = in_height
        self.in_depth = in_depth
        self.n_class = n_class
        self.out_dim = out_dim
        self.backbone=res_base_model

        print(self.in_width, self.in_height, self.in_depth)
        
        self.ct_features, self.ct_encoder ,self.ct_attention= self._init_resnet(pretrain_path)
        self.cta_features, self.cta_encoder,self.cta_attention = self._init_resnet(pretrain_path)
        self.bert_model = self._get_bert_basemodel(bert_base_model, freeze_layers)
        
        self.bert_l1 = nn.Linear(768, 768)  # 768 is the size of the BERT embbedings
        self.bert_l2 = nn.Linear(768, out_dim)  # 768 is the size of the BERT embbedings


    def _init_resnet(self, pretrain_path):
        if self.backbone=="resnet50":
            resnet = resnet50(
                sample_input_W=self.in_width,
                sample_input_H=self.in_height,
                sample_input_D=self.in_depth,
                shortcut_type='B',
                no_cuda=False,
                num_seg_classes=self.n_class
            )
            num_ftrs = 2048#resnet50
        elif self.backbone=="resnet34":
            resnet = resnet34(
                sample_input_W=self.in_width,
                sample_input_H=self.in_height,
                sample_input_D=self.in_depth,
                shortcut_type='B',
                no_cuda=False,
                num_seg_classes=self.n_class
            )
            num_ftrs=512#resnet34
        
        elif self.backbone=="resnet18":
            resnet= resnet18(
            sample_input_W=self.in_width,
            sample_input_H=self.in_height,
            sample_input_D=self.in_depth,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=self.n_class
        )
            resnet = add_dropout_to_resnet(resnet, dropout_rate=0.5)
            num_ftrs=512
        
        
        
        res_features = nn.Sequential(*list(resnet.children())[:-1])
        res_attention= CrossModalAttention3D(
            in_channels=num_ftrs,
            mag_channels=1,
            num_heads=8,
            # dropout=0.1
        )
        res_encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, self.out_dim),
        )
        
        
        return res_features, res_encoder,res_attention

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("Image feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")
        else:
            print('success')

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    def mean_pooling(self, model_output, attention_mask):
        
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def ct_encode(self, ct):
        f = self.ct_features(ct)
         
        mag,_=self.tofft(ct)
        f1=self.ct_attention(f,mag)
        x = self.ct_encoder(f1)

        return f, x
    
    def cta_encode(self, cta):
        f = self.cta_features(cta)
        mag,_=self.tofft(cta)
        f1=self.cta_attention(f,mag)
        x = self.cta_encoder(f1)

        return f, x

    
    def text_encoder(self, tokens):
        
        outputs = self.bert_model(
            
            **tokens
        )
        
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, tokens['attention_mask']).float()
            x = self.bert_l1(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2(x)

        return out_emb
    
    def tofft(self,ct):
        
        #ct->fft, coming soon
        


        return fft_ct_mag,fft_ct_phase
    


    def forward(self, ct, cta, tokens):
        _, ct_frt = self.ct_encode(ct)
        _, cta_frt = self.cta_encode(cta)
        text_frt = self.text_encoder(tokens)

        cos_sim_ct_cta = F.cosine_similarity(ct_frt, cta_frt)
        cos_sim_ct_text = F.cosine_similarity(ct_frt, text_frt)
        cos_sim_cta_text = F.cosine_similarity(cta_frt, text_frt)

        return ct_frt, cta_frt, text_frt,cos_sim_ct_cta, cos_sim_ct_text, cos_sim_cta_text

