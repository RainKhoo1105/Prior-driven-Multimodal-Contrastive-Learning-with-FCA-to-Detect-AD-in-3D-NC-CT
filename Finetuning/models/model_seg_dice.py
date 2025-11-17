"""
Reference for BERT Sentence Embeddings method

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",

"""


import torch.nn as nn
import torch.nn.functional as F
import torch
from models.resnet_seg_dice import resnet50

# Create the BertClassfier class
class ModelCLR(nn.Module):
    def __init__(
        self, in_width, in_height, in_depth, res_base_model, n_class,
        bert_base_model, out_dim, freeze_layers, do_lower_case, pretrain_path=None,freeze_encoder=False
    ):
        super(ModelCLR, self).__init__()
        self.in_width = in_width
        self.in_height = in_height
        self.in_depth = in_depth
        self.n_class = n_class
        self.out_dim = out_dim
        
        self.ct_resnet = self._init_resnet(pretrain_path,freeze_encoder)
   

    def _init_resnet(self, pretrain_path,freeze_encoder=False):
        resnet = resnet50(
            sample_input_W=self.in_width,
            sample_input_H=self.in_height,
            sample_input_D=self.in_depth,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=self.n_class
        )
        num_ftrs = 2048#resnet50

        # resnet= resnet.resnet34(
        #     sample_input_W = self.in_width,
        #     sample_input_H = self.in_height,
        #     sample_input_D = self.in_depth,
        #     shortcut_type='B',
        #     no_cuda=False,
        #     num_seg_classes=self.n_class
        # )
        # num_ftrs=512#resnet34

        net_dict = resnet.state_dict()
        if pretrain_path is not None:
            print("pretrain_path:",pretrain_path)
            pretrain = torch.load(pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
        resnet.load_state_dict(net_dict)
        
        if freeze_encoder:
            print("freeze encoder")
            for param in resnet.parameters():
                param.requires_grad = False  
            for param in resnet.decoder.parameters():
                param.requires_grad = True  
        
        return resnet

    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    
    def forward(self, ct):
        ct_seg=self.ct_resnet.forward(ct)

        return ct_seg


