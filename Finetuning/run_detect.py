import os
import torch
import logging
import datetime

from pathlib import Path
from utils import *
from tqdm import tqdm
from pprint import pprint
from omegaconf import OmegaConf
from models.ctanet.ctanet_ffl_attention import CTANet
from datasets.newdatasets import *
# from datasets.dataset3d import *
from torch.utils.data import DataLoader


os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]  = '1'
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class Trainer:
    def __init__(
        self, yaml_path, save_root, train_split_rate=0.8, 
        device_ids=[0, 1], batch_size=2, stop_type='acc', 
        epoch_num=50, lr=1e-5, seg_rate=0.5, save_per_epoch=5, 
        pretrain_path=None,tag='resnet50',width=64, freeze=True,unfreeze_num=0,modelname='ctctatext',datasetsname='xxxx',
        **kwargs
        
    ):
        self.date = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.freeze = freeze
        self.modelname=modelname
        self.save_root = save_root
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        print('lr:',self.lr)
        self.seg_rate = seg_rate
        self.stop_type = stop_type
        self.save_per_epoch = save_per_epoch
        self.seed=torch.initial_seed()
        self.datasetsname=datasetsname
        self.testbs=1
        self.nclass=2
        logging.basicConfig(filename=f'logs/{self.datasetsname}/{self.date}_{self.modelname}_epoch={self.epoch_num}_seed={self.seed}_nclass={self.nclass}_{self.freeze}freeze.log', level=logging.DEBUG)
        self.logger = logging.getLogger()
        config = OmegaConf.load(yaml_path)
        config_dict = OmegaConf.to_container(config, resolve=True)
        print('###NOW LOADING DATASET###')

        train_set = CTADataset(datatype='train', **config_dict['dataset'])
        valid_set = CTADataset(datatype='valid', **config_dict['dataset'])
        
        
        print('###NOW INITING MODEL###')
        
        self.net = CTANet(
            in_depth=config_dict['dataset']['depth'],
            in_width=width,
            in_height=width,
            n_class=self.nclass,
            gpu_id=device_ids,
            pretrain_path=pretrain_path,
            freeze=freeze,unfreeze_num=unfreeze_num,tag=tag
        ).cuda()
        
        if len(device_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])
            
        print('###NOW SPLITING DATASET###')
        print("seed:",self.seed)
        self.logger.info(f'***************************************')
        self.logger.info(f"seed:{self.seed}")
        self.logger.info(f'modelname:{self.modelname}')
        self.logger.info('***************************************')

        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, 
            drop_last=True, shuffle=True, num_workers=4
        )
        self.valid_loader = DataLoader(
            valid_set, batch_size=batch_size, #valid batchsize=1
            drop_last=False, shuffle=False, num_workers=4
        )
       
        print(len(train_set), len(valid_set))
        self.logger.info(f'train len:{len(train_set)},valid len:{len(valid_set)}')
        del train_set, valid_set
        
        self.save_path=Path(os.path.join(self.save_root, f'{self.date}_{self.modelname}_seed-{self.seed}_nclass-{self.nclass}_batchsize-{self.batch_size}_lr-{self.lr}_freeze{self.freeze}_seg_rate-{self.seg_rate}'))
        
        
   

    def test(self,load_path,model_idx=None,bigidx=None,flag=None):
        all_y_true = [] 
        all_y_scores = []  
        all_pred_label=[]
        total=[]
        print("################################################")
        print("####### NOW TEST MODEL #######")
        self.logger.info("####### NOW TEST MODEL #######")
        if load_path=='train':
            load_path=self.save_path
        if model_idx is not None:
            
            
            pretrain_path=os.path.join(load_path, 'checkpoints', f'{model_idx}.pth')
            self.net.img_encoder = nn.Sequential(*list(self.net.img_encoder.children())[:-1]) 
            self.net.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cuda')))
         
            print("load model:",os.path.join(load_path, 'checkpoints', f'{model_idx}.pth'))
            self.logger.info(f"load model:{os.path.join(load_path, 'checkpoints', f'{model_idx}.pth')}")
        elif bigidx is not None:
            best_files=[]
            for filename in os.listdir(load_path):
                if filename.startswith("best_") and filename.endswith(".pth"):
                    num_part=filename[5:-4]
                    num = int(num_part)  
                    if num<=bigidx:
                        best_files.append((num, filename))
            if best_files:
                largest_num, largest_file_name = max(best_files)
            md = torch.load(os.path.join(load_path, largest_file_name))
            self.net.load_state_dict(md)
            print("load model:",os.path.join(load_path, largest_file_name))
            self.logger.info(f"load model:{os.path.join(load_path, largest_file_name)}")
        else:
            best_files=[]
            for filename in os.listdir(load_path):
                if filename.startswith("best_") and filename.endswith(".pth"):
                    num_part=filename[5:-4]
                    try:
                        num = int(num_part) 
                        best_files.append((num, filename))
                    except ValueError:
                        continue
            if best_files:
                largest_num, largest_file_name = max(best_files)
            md = torch.load(os.path.join(load_path, largest_file_name))
            print("load model:",os.path.join(load_path, largest_file_name))
            self.logger.info(f"load model:{os.path.join(load_path, largest_file_name)}")
            self.net.load_state_dict(md)
        
        
        loss_fn = nn.CrossEntropyLoss()#1:1
        pt, nt = 0, 0
        
        for info, ct, _, _, label,_ in self.valid_loader:
            for l in label:
                if l == 0:
                    nt += 1
                else:
                    pt += 1
    
        print(pt, nt)
        self.logger.info(f'{pt},{nt}')
        self.net.eval()
        test_loss = 0
        test_Acc = 0
        total_p = total_r = 0
        total_cm = [0, 0, 0, 0]



        with torch.no_grad():
            
            loop = tqdm(self.valid_loader, desc='Infering')
            
            
            error_infos=[]
            for info, ct, _, _, label,_ in loop:
                
                ct = torch.unsqueeze(ct.permute(0, 1, 2, 3), dim=1).cuda()
                label = label.cuda()
                pred_label = self.net(ct)
                
                loss = loss_fn(pred_label, label)
                test_loss += loss.item()
                pred= torch.argmax(torch.softmax(pred_label, dim=-1), dim=-1) # b,
                p, r, cm = Trainer.cal_confusion_matrix(pred, label)
                total_p += p
                total_r += r
                total_cm = [total_cm[i] + int(cm[i]) for i in range(4)]
                acc = round(len(torch.where(pred == label)[0])/self.batch_size, 4)
                test_Acc += acc

                incorrect_mask = (pred != label)
                if torch.any(incorrect_mask):
                    incorrect_indices = torch.where(incorrect_mask)[0].cpu().numpy()
                    for idx in incorrect_indices:
                        if idx < len(info):
                            error_info = info[idx].split("\t")[0]
                            error_infos.append(error_info)
                

                label_numpy = label.cpu().numpy()
                pred=pred.cpu().numpy()
                pred_prob = torch.softmax(pred_label, dim=-1)[:, 1].cpu().numpy()  
                all_y_true.extend(label_numpy)
                all_y_scores.extend(pred_prob)
                all_pred_label.extend(pred)
                
            
            total.append(all_pred_label)
            total.append(all_y_true)
            total.append(all_y_scores)
            np.save(f'./outModels/{self.datasetsname}/{self.modelname}.npy', np.array(total))

            
  
            print('####################################')
            print('print test_set result:')
            print('ACC:', test_Acc / len(self.valid_loader))
            print('Precision:', total_p / len(self.valid_loader))
            print('Recall:', total_r / len(self.valid_loader))
            print('Confusion Matrix:')
            print('  P  N')
            print('P', total_cm[0], total_cm[1])
            print('N', total_cm[3], total_cm[2])
            tp=total_cm[0]
            fp=total_cm[1]
            tn=total_cm[2]
            fn=total_cm[3]
            acc= (tp + tn) / (tp + fp + fn + tn)
            precision=tp / (tp + fp) if (tp + fp) > 0 else 0
            recall=tp / (tp + fn) if (tp + fn) > 0 else 0
            sens=tp/(tp+fn)if (tp + fn) > 0 else 0
            spec=tn/(tn+fp)if (tn + fp) > 0 else 0
            print('Only use confusion matrix to calculate:')
            print('ACC:',acc)
            print('Precision:', precision)
            print('Recall:',recall )
            print("Sensitivity:",sens)
            print("Specificity:",spec)
            print('F1 Score:', 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
            self.logger.info('print test_set result:')
            self.logger.info('Confusion Matrix:')
            self.logger.info('  P  N')
            self.logger.info(f'P {total_cm[0]} {total_cm[1]}')
            self.logger.info(f'N {total_cm[3]} {total_cm[2]}')
            self.logger.info('Only use confusion matrix to calculate:')
            self.logger.info(f'ACC:{acc}')
            self.logger.info(f'Precision:{precision}')
            self.logger.info(f'Recall:{recall} ')
            self.logger.info(f"Sensitivity:{sens}")
            self.logger.info(f"Specificity:{spec}")
            self.logger.info(f'F1 Score:{2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0}')
            self.logger.debug('-------------------------------------------------------------------------')
            print("error list:")
            for error_info in error_infos:
                print(error_info)
            self.logger.info("error list:")
            for error_info in error_infos:
                self.logger.info(error_info)


        return all_y_true,all_y_scores




    def update_config(self):
        pass
    
    @staticmethod
    def cal_confusion_matrix(pred, y):
        # print(pred, y)
        TP = len(torch.where((pred == 1) & (y == 1))[0])
        FP = len(torch.where((pred == 1) & (y == 0))[0])
        TN = len(torch.where((pred == 0) & (y == 0))[0])
        FN = len(torch.where((pred == 0) & (y == 1))[0])
        Precision = TP / (TP + FP) if TP > 0 else 0
        Recall = TP / (TP + FN) if TP > 0 else 0
        return Precision, Recall, [TP, FP, TN, FN]
    
    
    
    @staticmethod
    def set_seed():
        pass
        

if __name__ == '__main__':
    trainer = Trainer(
        yaml_path='configs/config3d_newsets_ffl_attention.yaml',
        save_root=f'save path',
        batch_size=64,
        device_ids=[2],
        epoch_num=200,
        lr=1e-5,
        seg_rate=0.8,
        pretrain_path=f'/Pretrain/runs/0902_resnet50_fftloss2_attention_wrf91/checkpoints/xx.pth',
        freeze=True,
        unfreeze_num=2,
        tag='resnet50',
        width=80,
        modelname=f'xxx',
        datasetsname=f'xxx',# dir name
    )
    
    trainer.test(load_path="./outModels/Classification/xx.pth",model_idx=99)
    