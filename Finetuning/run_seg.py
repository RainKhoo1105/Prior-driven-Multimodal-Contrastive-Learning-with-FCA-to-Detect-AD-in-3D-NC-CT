import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.dataset3d_seg_dice import *
from torch.utils.data import DataLoader
import torch.nn as nn
import SimpleITK as sitk
import yaml

from models.model_seg_dice import ModelCLR

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = inputs.float()
        targets = targets.float()
        intersection = torch.sum(inputs * targets, dim=(0,2,3,4)) 
        union = torch.sum(inputs, dim=(0,2,3,4)) + torch.sum(targets, dim=(0,2,3,4))
        dice = (2. * intersection + smooth) / (union + smooth) 
        loss = 1 - torch.mean(dice)
        return loss
    

    

class SimCLR(object):
    def __init__(self, config, gpu_id=[0], dataset=None,modelname='ct'):
        self.date = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.config = config
        self.cross_dice_weight=config['cross_dice_weight']
        print(f"cross_dice_weight:cross {self.cross_dice_weight[0]}:dice {self.cross_dice_weight[1]}")
        self.gpu_id = gpu_id
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.seed=torch.initial_seed()
        self.modelname=modelname
        self.do_bg=config['do_bg']#true means calculate background dice loss, false means only calculate foreground dice loss
        ###############
        train_set = CTADataset(datatype='train', **config['dataset'])
        valid_set = CTADataset(datatype='valid', **config['dataset'])
        print(f'modelname=',self.modelname,',seed=',self.seed,',date=',self.date)
        print(len(train_set), len(valid_set))
        print(' lr=',config['learning_rate'])
        
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
       
        self.truncation = config["truncation"]
        
    def _get_device(self):
        device = f"cuda:{self.gpu_id[0]}" if torch.cuda.is_available() else "cpu"
        print("Running on:", device)
        return device

    
    def test(self,load_path="path of segmentation model weights",idx='100'):
        def save_as_nii_gz(pred, info, result_dir,output_path,flag):
            
            name=info.split("/")[-1]
            reference_path=''
            if flag=='m':
                reference_path = os.path.join(info,f"straightenmask_{name}.nii.gz")
            elif flag=='i':
                reference_path = os.path.join(info,f"straightencta_{name}.nii.gz")
                
            if not os.path.exists(reference_path):
                print(f"Reference file not found: {reference_path}")
                return
            ref_image = sitk.ReadImage(reference_path)
            origin = ref_image.GetOrigin()
            spacing = ref_image.GetSpacing()
            direction = ref_image.GetDirection()
            
            if flag=='m':
                pred_image = sitk.GetImageFromArray(pred.astype(np.uint8))
            else:
                pred_image = sitk.GetImageFromArray(pred.astype(np.float32))
            pred_image.SetOrigin(origin)
            pred_image.SetSpacing(spacing)
            pred_image.SetDirection(direction)
            sitk.WriteImage(pred_image, os.path.join(result_dir, output_path))

                
               
        model = ModelCLR(**self.config["model"]).to(self.device)
        
        checkpoint = torch.load(os.path.join(load_path, 'checkpoints', f'{idx}.pth'), map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        
        print("load model:",os.path.join(load_path, 'checkpoints', f'{idx}.pth'))
        model.eval()
        result_dir = f'xx/{self.modelname}/nii' # path of result
        os.makedirs(result_dir, exist_ok=True)
        
        
        with torch.no_grad():
            loop = tqdm(self.valid_loader, desc='Infering')
            for info, ct, cta,_, _,mask in loop:
                if self.do_bg:
                    ct_tensor=torch.unsqueeze(ct.permute(0,1,2,3),dim=1).to(self.device).to(torch.float32)
                    mask_tensor=mask.to(self.device).to(torch.float32).cpu().numpy()
                    cta_tensor=cta.to(self.device).to(torch.float32).cpu().numpy()
                    ct_np=ct.to(self.device).to(torch.float32).cpu().numpy()

                    ct_seg= model(ct_tensor)
                    pred_sample = torch.argmax(ct_seg, dim=1).cpu().numpy()
                    
                else:
                    mask1=torch.unsqueeze(mask.permute(0,1,2,3),dim=1)
                    mask1 = mask1.to(self.device).to(torch.float32)
                    one_hot_mask = torch.nn.functional.one_hot(
                        mask1.squeeze(1).long(), 
                        num_classes=3
                    ).permute(0, 4, 1, 2, 3).float()  

                    ct_tensor=torch.unsqueeze(ct.permute(0,1,2,3),dim=1).to(self.device).to(torch.float32)
                    mask_tensor=mask.to(self.device).to(torch.float32).cpu().numpy()
                    cta_tensor=cta.to(self.device).to(torch.float32).cpu().numpy()
                    ct_np=ct.to(self.device).to(torch.float32).cpu().numpy()

                    ct_seg= model(ct_tensor)
                    

                    mask_aorta = (mask > 0) 
                    mask_aorta_oh = mask_aorta.unsqueeze(1).repeat(1,3,1,1,1)
                    if mask_aorta.sum() > 0:
                        mask_aorta_oh=mask_aorta_oh.to(self.device)

                        ct_seg = ct_seg * mask_aorta_oh

                    pred_sample =torch.argmax(ct_seg,dim=1) 
                    
                    one_hot_pred_sample = torch.nn.functional.one_hot(
                        pred_sample,
                        num_classes=3
                    ).permute(0, 4, 1, 2, 3).float()  
                    one_hot_pred_sample = one_hot_pred_sample[:,1:,:,:,:]
                    one_hot_mask = one_hot_mask[:,1:,:,:,:]
                    
                    intersection = torch.sum(one_hot_pred_sample * one_hot_mask, dim=(2,3,4)) 
                    union = torch.sum(one_hot_pred_sample, dim=(2,3,4)) + torch.sum(one_hot_mask, dim=(2,3,4))  

                    
                    smooth = 1e-5
                    dice = (2. * intersection + smooth) / (union + smooth)  
                    seg_dice = torch.mean(dice)
                    print("seg dice:",seg_dice)
                    pred_sample = pred_sample.cpu().numpy()
                    
                    ct_seg = ct_seg.cpu().numpy()
                    
                for i in range(pred_sample.shape[0]):
                    infoi=info[i].split('\t')[0]
                    sample_info = info[i].split('\t')[0].split('/')[-1]
                    
                    save_as_nii_gz(pred_sample[i],infoi,result_dir,f"{sample_info}_pred.nii.gz",'m')
                    save_as_nii_gz(ct_np[i],infoi,result_dir,f"{sample_info}_ct.nii.gz",'i')
                    save_as_nii_gz(cta_tensor[i],infoi,result_dir,f"{sample_info}_cta.nii.gz",'i')
                    save_as_nii_gz(mask_tensor[i],infoi,result_dir,f"{sample_info}_mask.nii.gz",'m')
if __name__ == '__main__':
    config = yaml.load(open("./configs/config_seg_dice.yaml", "r"), Loader=yaml.FullLoader)
    simclr = SimCLR(config, gpu_id=[2],modelname='xxxx')
    simclr.test("./outModels/Segmentation/xx.pth",'model')