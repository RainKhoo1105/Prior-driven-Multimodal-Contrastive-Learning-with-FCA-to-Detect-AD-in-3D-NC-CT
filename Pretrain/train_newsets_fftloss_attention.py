

import logging
import os
import shutil

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets.newdatasets import *
from loss.nt_xent import NTXentLoss
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F  

from models.model_newsets_fftloss_attention import ModelCLR
from models.focal_frequency_loss import FocalFrequencyLoss as FFL

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(
            "./config_newsets_dataenhance.yaml", os.path.join(model_checkpoints_folder, "config_newsets_dataenhance.yaml")
        )


class SimCLR(object):
    def __init__(self, config, gpu_id=[0], dataset=None,modelname='ct',weight=[0.4,0.6]):
        self.date = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.config = config
        self.gpu_id = gpu_id
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.seed=torch.initial_seed()
        self.weight=weight
        self.fca_loss_weight=self.config["fca_loss_weight"]  
        self.alpha_weight=config["loss"]['alpha_weight']
        print("backbone=",config["model"]["res_base_model"])
        print(f"weight text:cta={self.weight[0]}:{self.weight[1]}")
        print(f"fca_loss_weight={self.fca_loss_weight[0]}:{self.fca_loss_weight[1]}")
        self.modelname=modelname
        ###############
        train_set = CTADataset(datatype='train', **config['dataset'])
        print("ct type:",train_set[0][1].dtype)
        valid_set = CTADataset(datatype='valid', **config['dataset'])
        print(f'modelname=',self.modelname,',seed=',self.seed,',date=',self.date)
        print("len train set",len(train_set),"len valid set", len(valid_set))
        print("lr=",self.config["learning_rate"])

        self.train_loader = DataLoader(
            train_set, batch_size=config['batch_size'], 
            drop_last=True, shuffle=True, num_workers=4
        )
        self.valid_loader = DataLoader(
            valid_set, batch_size=config['batch_size'], 
            drop_last=True, shuffle=True, num_workers=4
        )
        del train_set, valid_set
        ###############
        
        self.nt_xent_criterion = NTXentLoss(
            self.device, config["batch_size"], **config["loss"]
        )
        self.truncation = config["truncation"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["bert_base_model"]
        )  
        
    def _get_device(self):
        device = f"cuda:{self.gpu_id[0]}" if torch.cuda.is_available() else "cpu"
        print("Running on:", device)
        return device
    
    def freq_transform(self,x):
        
        x = x.float()
        
        x_fft = torch.fft.fft(x, dim=1)
        x_abs = torch.abs(x_fft) + 1e-8  
        x_angle=torch.angle(x_fft)
        
        x_log = torch.log(x_abs)
        
        x_norm = F.normalize(x_log, p=2, dim=1)
        return x_norm

    def train(self):
        
        if len(self.gpu_id) == 1:
            os.environ['CUDA_VISIBLE_DEVICE'] = str(self.gpu_id[0])
            model = ModelCLR(**self.config["model"]).to(self.device)
        else:
            model = DataParallel(
                ModelCLR(**self.config["model"]).to(self.device),
                device_ids=self.gpu_id
            )
            

        optimizer = torch.optim.Adam(
            model.parameters(),
            eval(self.config["learning_rate"]),
            weight_decay=eval(self.config["weight_decay"]),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1
        )

        scaler = GradScaler()

        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        
        
        for epoch_counter in range(self.config["epochs"]):
            total_loss = 0
            total_text = 0
            total_cta = 0
            total_cta_ft=0
            total_text_ft=0
            for _, ct, cta, text, _ ,_ in tqdm(self.train_loader):
                optimizer.zero_grad()
                tokens = self.tokenizer(
                    list(text),
                    return_tensors="pt",
                    padding=True,
                    truncation=self.truncation,
                ).to(self.device)
                
                ct=torch.unsqueeze(ct.permute(0,1,2,3),dim=1)
                cta=torch.unsqueeze(cta.permute(0,1,2,3),dim=1)
                ct=ct.to(self.device)
                cta=cta.to(self.device)
            
                with autocast():
                    # calculate loss, coming
                    ct_frt, cta_frt, text_frt,cos_sim_ct_cta, cos_sim_ct_text, cos_sim_cta_text= model(ct, cta, tokens) 
                    

                if n_iter % self.config["log_every_n_steps"] == 0:
                    self.writer.add_scalar("train_loss", loss, global_step=n_iter)

                # Scales the loss to create scaled gradients
                scaler.scale(loss).backward()

                # Unscales the gradients
                scaler.step(optimizer)
                scaler.update()
                n_iter += 1

            print(f"Epoch {epoch_counter} ------ Train Loss: {total_loss / len(self.train_loader)}, cta loss {total_cta / len(self.train_loader)}, text loss {total_text / len(self.train_loader)}")
            print(f"cta_ft loss {total_cta_ft/len(self.train_loader)}, text_ft loss {total_text_ft/len(self.train_loader)}")

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                valid_loss = self._validate(model, self.valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoints_folder, "model.pth"),
                    )
                if epoch_counter % 100 == 0:
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, f'{epoch_counter}.pth'))
                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1
                print(f"Validation {epoch_counter} - Valid Loss: {valid_loss}")

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step(valid_loss)
            self.writer.add_scalar(
                "cosine_lr_decay",
                scheduler.get_last_lr()[0],
                global_step=n_iter,
            )

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./runs", self.config["fine_tune_from"], "checkpoints"
            )
            state_dict = torch.load(os.path.join(checkpoints_folder, "model.pth"))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_iter):
        # validation steps
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            for _, ct, cta, text, _ ,_ in tqdm(valid_loader):
                tokens = self.tokenizer(
                    list(text),
                    return_tensors="pt",
                    padding=True,
                    truncation=self.truncation,
                ).to(self.device)
                ct=torch.unsqueeze(ct.permute(0,1,2,3),dim=1)
                cta=torch.unsqueeze(cta.permute(0,1,2,3),dim=1)
                ct=ct.to(self.device)
                cta=cta.to(self.device)
                # calculate loss, coming
                
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
