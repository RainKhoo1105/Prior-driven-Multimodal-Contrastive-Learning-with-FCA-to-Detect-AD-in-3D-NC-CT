
import sys
sys.path.append('../')
import torch
import cv2
import os
# import docx
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
import datetime
import json
import logging
import random

from scipy.ndimage import zoom
from tqdm import tqdm
from joblib import Parallel, delayed
from pprint import pprint
from pathlib import Path
from torch.utils.data import Dataset
from scipy.ndimage import rotate


class DatasetLoader:
    def __init__(
        self, path,save_path, threshold, save_verbose=False, 
        train_ratio=0.8, trans_dict_path='./', resolution=[512, 512], 
        discard=5, roi=False, text_mode='all', timestamp=None, depth=512
    ):
        '''
        Params:
            path: data path, support multiple input
            match_path: csv path, support multiple input
            save_path: save path
            threshold: CT and CTA image threshold
            save_verbose: whether to visualize dataset
            train_ratio: train ratio
            resolution: default image resolution
            text_mode: text process mode, now support two: 1. keep all diagnosis report; 2. use jieba to get keywords
            timestamp: dataset timestamp, if specified, load dataset at that timestamp, otherwise load latest dataset
            depth: resample after 3D image depth
        '''
        self.paths = list(map(Path, path)) if isinstance(path, list) else [Path(path)]
        self.save_path = Path(save_path)
        self.threshold = threshold
        self.save_verbose = save_verbose
        self.train_ratio = train_ratio
        self.resolution = tuple(resolution)
        self.discard = discard
        self.roi = roi
        assert text_mode in ['all', 'jieba']
        self.text_mode = text_mode
        self.timestamp = timestamp
        self.depth = depth
        self.notext_list = []
        self.noct_list = []
        with open(trans_dict_path, 'r') as f:
            self.trans_dict = json.load(f)
        self.data_pos = []#datapos
        self.data_neg = []#dataneg
        self.load_data()
        train_pos, valid_pos = self.split_data(self.data_pos)#,train_size=283
        train_neg, valid_neg = self.split_data(self.data_neg)#,train_size=259
        self.train, self.valid = train_pos, valid_pos
        self.train.extend(train_neg)
        self.valid.extend(valid_neg)
        self.save_data()
            
    def load_data(self):            
        def load_parallel(dir):
            try:
                CT_slices, CTA_slices,seg,up= self.read_data(dir)
                # print('before',CT_slices.shape)
                up_size=up.shape[0]
                up_img=seg[:up_size,:,:]
                down_img=seg[up_size:,:,:]
            except Exception as e:
                print(repr(e))
            else:
                CT_slices  = self.transform(CT_slices, self.threshold['ct'], 'i')
                CTA_slices = self.transform(CTA_slices, self.threshold['cta'], 'i')
                seg,up,down= self.transform(seg, None, 'm',up_img,down_img)
                if (up.shape[0]+down.shape[0])!=seg.shape[0]:
                    print("length is false")
                 
                if self.roi:
                    mask_array = np.zeros_like(seg)
                    mask_array[seg > 0] = 1
                    CT_slices *= mask_array
                    CTA_slices *= mask_array
                 
                filename = str(dir).split('/')[-1]
                if CT_slices is not None and CTA_slices is not None:
                     
                    if filename in self.trans_dict.keys():
                        text, label = self.trans_dict[filename]['text'], self.trans_dict[filename]['label']
                    else:
                        print('no text!')
                        self.notext_list.append(filename)
                    
                    CT_slices=CT_slices.astype(np.float32)
                    CTA_slices=CTA_slices.astype(np.float32)
                    #zyx-> xyz
                    CT_slices= np.transpose(CT_slices, (2, 1, 0))
                    CTA_slices = np.transpose(CTA_slices, (2, 1, 0))
                    seg=np.transpose(seg, (2, 1, 0))
                    up=np.transpose(up, (2, 1, 0))
                    down=np.transpose(down, (2, 1, 0))

                     
                    if text is not None:
                        fulltext=text
                        text = text if len(text) < 512 else text[:512]
                        # print(label)
                        if label == 1 or label ==2:
                            self.data_pos.append(
                                [
                                    f'{str(dir)}\t{CT_slices.shape[2]}', 
                                    CT_slices,CTA_slices,text,label,
                                    seg,up,down,fulltext
                                ]
                            )
                        elif label ==0:
                            self.data_neg.append(
                                [
                                    f'{str(dir)}\t{CT_slices.shape[2]}', 
                                    CT_slices,CTA_slices,text,label,
                                    seg,up,down,fulltext
                                ]
                            )

                       
                else:
                    print('no CT!')
                    self.noct_list.append(filename)
        dirs = []
        for path in self.paths:
            for item in path.iterdir():
                if item.is_dir():
                    dirs.append(item)
        print('----TOTALLY GET {} DIRS----'.format(len(dirs)))
        # print(dirs)
        Parallel(n_jobs=4, backend='threading')(delayed(load_parallel)(
            dir
        ) for dir in tqdm(dirs))
    
    def split_data(self, data,train_size=None):
        if train_size==None:
            print(f'----NOW SPLITING DATASET WITH RATIO {self.train_ratio}----')
            random.seed(42)
            random.shuffle(data)
            print("split data all:",len(data))
            self.train, self.valid = [], []
            train_size = int(len(data) * self.train_ratio)+1
        train, valid = data[:train_size], data[train_size:]
        print('trainset:', len(train), 'validset:', len(valid))
        return train, valid
         
    def save_data(self):
        print('---DATASET GENERATED, NOW SAVING----')
        date = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(date)
        threshold = '_'.join([f'{a[0]}:{a[1]}' for a in self.threshold.values()])
        data_name = f'{date}_discard-{self.discard}_threshold{threshold}_roi-{self.roi}_depth{self.depth}_textmode-{self.text_mode}'
        data_path = self.save_path / data_name
        data_path.mkdir(exist_ok=True)
        self.train = np.asarray(self.train, dtype=object)
        self.valid = np.asarray(self.valid, dtype=object)
        np.save(str(data_path / 'train.npy'), self.train, allow_pickle=True)
        np.save(str(data_path / 'valid.npy'), self.valid, allow_pickle=True)
        if self.save_verbose:
            (data_path / 'CT').mkdir(exist_ok=True)
            (data_path / 'CTA').mkdir(exist_ok=True)
            (data_path / 'mask').mkdir(exist_ok=True)
            info = {}
            def save_parallel(data_slice):
                filename = data_slice[0].split('\t')[0].split('/')[-1]
                for idx in range(20,25):
                    img_name = f'{filename}-{idx}.jpg'
                    cv2.imwrite(str(data_path / 'CT' / img_name), data_slice[1][:,:, idx] * 255)
                    cv2.imwrite(str(data_path / 'CTA' / img_name), data_slice[2][:,:, idx] * 255)
                    cv2.imwrite(str(data_path / 'mask' / img_name), data_slice[5][:,:, idx] * 127.5)
                    
                if filename not in info.keys():
                    info[filename] = {
                        'text': str(data_slice[3]),
                        'label': int(data_slice[4])
                    }
            Parallel(n_jobs=16, backend='threading')(delayed(save_parallel)(
                data_slice
            ) for data_slice in tqdm(self.train))
            Parallel(n_jobs=16, backend='threading')(delayed(save_parallel)(
                data_slice
            ) for data_slice in tqdm(self.valid))
            with open(data_path / 'info.json', 'w') as f:
                json.dump(info, f, ensure_ascii=False)
            with open(data_path / 'err_log.txt', 'w') as f:
                f.write('--------no ct file--------\n')
                for name in self.noct_list:
                    f.write(name + '\n')
                f.write('\n--------no text file--------\n')
                for name in self.notext_list:
                    f.write(name + '\n')
    
    def read_data(self, path):
        ct = cta = seg= None
        try: 
            name = str(path).split('/')[-1]
             
            ct_path = os.path.join(path, 'straightenct_'+name+'.nii.gz')
            cta_path = os.path.join(path, 'straightencta_'+name+'.nii.gz')
            mask_path = os.path.join(path, 'straightenmask_'+name+'.nii.gz')
            up_path=os.path.join(path,'upmask_'+name+'.nii.gz')
            # print('1',ct_path)
            ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))[:,8:72,8:72].astype(np.float32)
            cta = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))[:,8:72,8:72].astype(np.float32)
            seg = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))[:,8:72,8:72]
            up=sitk.GetArrayFromImage(sitk.ReadImage(up_path))[:,8:72,8:72]
             
        except FileNotFoundError:
            self.noct_list.append(str(path))
            pass
        except AssertionError:
            pass
             
        except Exception as e:
            print(repr(e))
            
         
        return ct, cta, seg,up
                
    def transform(self, img: np.array, threshold=None, img_flag='i',up_img=None,down_img=None):
        try:
            scale = img.shape[0] / self.depth 
            if img_flag=='i':
                img = self.resample_image(img, [1, 1, 1], [scale, 1, 1], img_flag)
                assert threshold is not None, 'Find NoneType for threshold!'
                img = self.threshold_and_normalize(img, threshold[0], threshold[1])
                return img
            else:
                img,upvol,downvol= self.resample_image(img, [1, 1, 1], [scale, 1, 1], img_flag,up_img,down_img)
                return img,upvol,downvol
        except Exception as e:
            return None
    
    def resample_image(self, image, spacing, new_spacing, im_flag,up_img=None,down_img=None):
        assert (im_flag=='i' or im_flag=='m')
        shape = np.array(image.shape)
        newSize = np.round(shape * spacing / new_spacing)
        resize_factor = newSize / shape
        resize_factor = resize_factor.tolist()
         
        if im_flag == 'i':
            newVol = zoom(image, resize_factor, mode='nearest', order=2)
            return newVol
        else:
            newVol = zoom(image, resize_factor, mode='constant', order=0)
            upvol=zoom(up_img,resize_factor,mode='constant', order=0)
            downvol=zoom(down_img,resize_factor,mode='constant', order=0)
            return newVol,upvol,downvol
        
    
    def threshold_and_normalize(self, img, min_bound, max_bound):
        img[img < min_bound] = min_bound
        img[img > max_bound] = max_bound
        
        return (img - min_bound) / (max_bound - min_bound)
    

class CTADataset(Dataset):
    def __init__(self, load_path, datatype, discard, threshold, roi, text_mode, depth, timestamp=None):
        super().__init__()
        self.load_path = Path(load_path)
        self.datatype = datatype
        self.discard = discard
        self.threshold = threshold
        self.roi = roi
        self.text_mode = text_mode
        self.depth = depth
        self.timestamp = timestamp
        self.load_data()
        
    def load_data(self):
        try:
            if self.timestamp is None:
                threshold = '_'.join([f'{a[0]}:{a[1]}' for a in self.threshold.values()])
                data_name = f'discard-{self.discard}_threshold{threshold}_roi-{self.roi}_depth{self.depth}_textmode-{self.text_mode}'
                datasets = [dir for dir in list(self.load_path.glob('*')) if data_name in str(dir)]
                datasets.sort(key=lambda x: str(x).split('_')[0])
                path = datasets[-1]
                print(path)
            else:
                threshold = '_'.join([f'{a[0]}:{a[1]}' for a in self.threshold.values()])
                path = self.load_path / f'{self.timestamp}_discard-{self.discard}_threshold{threshold}_roi-{self.roi}_depth{self.depth}_textmode-{self.text_mode}'
                print(path)
            self.data = np.load(path / f'{self.datatype}.npy', allow_pickle=True)
            
        except Exception as e:
            print(repr(e))
        else:
            return
    def image_augmentation_3d(self,ct, cta,seg):
        I = ct.copy()
        G = cta.copy()
        K=seg.copy()

        angle = random.uniform(0, 360)
       
        I = rotate(I, angle, mode='nearest', axes=(0, 1), reshape=False)
        G = rotate(G, angle, mode='nearest', axes=(0, 1), reshape=False)
        K = rotate(K, angle, mode='nearest', axes=(0, 1), reshape=False)

        
        if random.random() > 0.5:
            I = np.flip(I, axis=2)
            G = np.flip(G, axis=2)
            K = np.flip(K, axis=2)

        
        if random.random() > 0.5:
            I = np.flip(I, axis=0)
            G = np.flip(G, axis=0)
            K = np.flip(K, axis=0)

        
        if random.random() > 0.5:
            I = np.flip(I, axis=1)
            G = np.flip(G, axis=1)
            K = np.flip(K, axis=1)
        
        I = I.astype(np.float32)
        G = G.astype(np.float32)
        K = K.astype(np.int64)

        return I, G ,K# ct,cta,seg
    
    def enhance(self,CT_slices,CTA_slices,seg,label,text):
        rotations = [90, 180, 270]
        for angle in rotations:
            rotated_ct = rotate(CT_slices, angle, axes=(0, 1), reshape=False)
            rotated_cta = rotate(CTA_slices, angle, axes=(0, 1), reshape=False)
            rotate_seg=rotate(seg,angle,axes=(0,1),reshape=False)
            if label == 1 or label==2:
                self.data_pos.append(
                    [
                    f'{str(dir)}\t{rotated_ct.shape[2]}',
                    rotated_ct, rotated_cta, text, label,rotate_seg
                    ])
            elif label == 0:
                self.data_neg.append(
                    [
                        f'{str(dir)}\t{rotated_ct.shape[2]}',
                        rotated_ct, rotated_cta, text, label,rotate_seg
                    ])
            
        
        flipped_ct = np.flip(CT_slices, axis=2)
        flipped_cta = np.flip(CTA_slices, axis=2)
        flipped_seg=np.flip(seg,axis=2)
        if label == 1 or label==2:
            self.data_pos.append(
                [
                    f'{str(dir)}\t{rotated_ct.shape[2]}',
                    flipped_ct, flipped_cta, text, label,flipped_seg
                ])
        elif label == 0:
            self.data_neg.append(
                [
                    f'{str(dir)}\t{rotated_ct.shape[2]}',
                    flipped_ct, flipped_cta, text, label,flipped_seg
                ])
    def __getitem__(self, idx):
        
         
        CT_arry   = self.data[idx][1].astype(np.float32)
        CTA_array = self.data[idx][2].astype(np.float32)
        seg=self.data[idx][5].astype(np.int64)
        label     = int(self.data[idx][4])
        if label!= 0:
            label =1

        CT_arry,CTA_array,seg= self.image_augmentation_3d(CT_arry,CTA_array,seg)
        CT_arry = np.ascontiguousarray(CT_arry)
        CTA_array = np.ascontiguousarray(CTA_array)
        seg=np.ascontiguousarray(seg)

        CT_arry = CT_arry.astype(np.float32)
        CTA_array = CTA_array.astype(np.float32)
        seg=seg.astype(np.int64)

        return self.data[idx][0], CT_arry, CTA_array, \
            self.data[idx][3],label, seg
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    loader = DatasetLoader(
        
        path=[
            '/xx/FirstBatch_straighten', # straighten aorta
            '......',
        ],
        
        save_path='xx/datasets',
        threshold={
          'ct': [0, 300],
          'cta': [0, 800]
        },
        train_ratio=0.8,
        trans_dict_path='xx/trans_dict_Batch123_3label.json',
        
        save_verbose=True,
        roi=False,
        discard=None,
        text_mode='all',
        depth=128,
        resolution=[64, 64]
    )
