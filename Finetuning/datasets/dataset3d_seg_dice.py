import sys
sys.path.append('../')

import cv2
import os
import docx
import nibabel as nib
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
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
from .sparkAPI import translate


logging.getLogger('jieba').setLevel(logging.CRITICAL)
# Spark Max URL，other URL please goto（https://www.xfyun.cn/doc/spark/Web.html）to check
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
SPARKAI_APP_ID = 'app id'#need to be replaced
SPARKAI_API_SECRET = 'api secret'#need to be replaced
SPARKAI_API_KEY = 'api key'#need to be replaced
SPARKAI_DOMAIN = '4.0Ultra'

# 2024-11-30-23:04:32
class DatasetLoader:
    def __init__(
        self, path, match_path, save_path, threshold, save_verbose=False, 
        train_ratio=0.8, trans_dict_path='./', clip_path='./', resolution=[512, 512], 
        discard=5, roi=False, text_mode='all', timestamp=None, depth=512
    ):
        '''
        Params:
            path: data path, support multiple inputs
            match_path: csv path of diagnosis report, support multiple inputs
            save_path: save path
            threshold: CT and CTA image threshold
            save_verbose: whether to visualize dataset
            train_ratio: train set ratio
            resolution: default image resolution
            text_mode: text processing mode, now support two modes: 1. keep all diagnosis report; 2. use jieba to get keywords
            timestamp: dataset timestamp, if specified, load dataset packed at that time, otherwise load the latest dataset
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
         
        assert isinstance(match_path, list)
        self.match_table = {}
        for mp in match_path:
            data = pd.read_excel(mp[0])
            self.match_table.update({
                data.iloc[i, 0]: [list(data.loc[i])[1:], mp[1]] for i in range(len(data))
            })
        with open(trans_dict_path, 'r') as f:
            self.trans_dict = json.load(f)
        
        self.data_pos = []
        self.data_neg = []
        self.load_data()
        train_pos, valid_pos = self.split_data(self.data_pos)
        train_neg, valid_neg = self.split_data(self.data_neg)
        self.train, self.valid = train_pos, valid_pos
        self.train.extend(train_neg)
        self.valid.extend(valid_neg)
        self.save_data()
            
    def load_data(self):            
        def load_parallel(dir):
            try:
                CT_slices, CTA_slices, text, label = self.read_data(dir)
            except Exception as e:
                print(repr(e))
            else:
                CT_slices = self.transform(CT_slices, self.threshold['ct'], 'i')
                CTA_slices = self.transform(CTA_slices, self.threshold['cta'], 'i')
                
                if CT_slices is not None and label is not None and CT_slices.shape[:2] == self.resolution:
                    
                    filename = str(dir).split('\t')[0].split('/')[-1]
                    if filename in self.trans_dict.keys():
                        text, label = self.trans_dict[filename]['text'], self.trans_dict[filename]['label']
                         
                    else:
                        text = translate(
                            appid=SPARKAI_APP_ID,
                            api_secret=SPARKAI_API_SECRET,
                            api_key=SPARKAI_API_KEY,
                            Spark_url = SPARKAI_URL,  # 4.0Ultra url
                            domain = SPARKAI_DOMAIN,  # 4.0Ultra domain
                            query=text
                        )
                    print('text:', text)
                    if text != '':
                        text = text if len(text) < 512 else text[:512]
                        print(label)
                        if label == 1:
                            self.data_pos.append(
                                [
                                    f'{str(dir)}\t{CT_slices.shape[-1]}', 
                                    CT_slices.transpose(2, 0, 1),
                                    CTA_slices.transpose(2, 0, 1),
                                     
                                    text, 
                                    label
                                ]
                            )
                        else:
                            self.data_neg.append(
                                [
                                    f'{str(dir)}\t{CT_slices.shape[-1]}', 
                                    CT_slices.transpose(2, 0, 1),
                                    CTA_slices.transpose(2, 0, 1),
                                     
                                    text, 
                                    label
                                ]
                            )
                 
        dirs = []
        for path in self.paths:
            for item in path.iterdir():
                if item.is_dir():
                    dirs.append(item)
        # for test
        dirs = dirs[:4] #############
        print('----TOTALLY GET {} DIRS----'.format(len(dirs)))
        Parallel(n_jobs=1, backend='threading')(delayed(load_parallel)(
            dir
        ) for dir in tqdm(dirs))
    
    def split_data(self, data):
        print(f'----NOW SPLITING DATASET WITH RATIO {self.train_ratio}----')
        random.seed(1)
        random.shuffle(data)
        print(len(data))
        self.train, self.valid = [], []
        train_size = int(len(data) * self.train_ratio) + 1
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
             
            info = {}
            def save_parallel(data_slice):
                filename = data_slice[0].split('\t')[0].split('/')[-1]
                for idx in range(self.depth):
                    img_name = f'{filename}-{idx}.jpg'
                    cv2.imwrite(str(data_path / 'CT' / img_name), data_slice[1][:, :, idx] * 255)
                    cv2.imwrite(str(data_path / 'CTA' / img_name), data_slice[2][:, :, idx] * 255)
                    
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
        CT_slices = CTA_slices = text = masks = label = None
        flag = False
        try:
            CT = nib.load(path / 'CT.nii')
            CTA = nib.load(path / 'CTA_register.nii')
             
            
            diagnosis_files = list(path.glob(r'*.docx'))
        except FileNotFoundError:
            self.noct_list.append(str(path))
            pass
        except AssertionError:
            pass
             
        except Exception as e:
            print(repr(e))
        else:
            CT_slices = np.array(CT.dataobj, dtype=np.float32)[:, :, self.discard: -self.discard]
            CTA_slices = np.array(CTA.dataobj, dtype=np.float32)[:, :, self.discard: -self.discard]
             
                
            if len(diagnosis_files) > 0:
                assert len(diagnosis_files) == 1
                document = docx.Document(path / diagnosis_files[0])
                name = str(path)
            else:
                name = document = str(path)
            text, label = self.read_docx(name, document)
            
        return CT_slices, CTA_slices, text, label
                
    def read_docx(self, filename, document):
         
        def pretreat(text):
            if text == '' or self.text_mode == 'all':
                return text
            elif self.text_mode == 'jieba':
                tags = jieba.analyse.extract_tags(
                    text, topK=50, withWeight=False, allowPOS=()
                )
                return ', '.join(tags)
            else:
                pass
        
        name = filename.split('/')[-1]
        text, label = '', None
        if name in self.match_table.keys():
            text, label = self.match_table[name]
            if isinstance(text, list):
                text = text[0]
            return pretreat(text), label
        else:
            self.notext_list.append(filename)
            return None, None
    
    def transform(self, img: np.array, threshold=None, img_flag='i'):
        try:
            for i in range(img.shape[-1]):
                img[:, :, i] = img[:, :, i].T
            
            if img_flag == 'i':
                assert threshold is not None, 'Find NoneType for threshold!'
                img = self.threshold_and_normalize(img, threshold[0], threshold[1])
            
            scale = img.shape[-1] / self.depth
            return self.resample_image(img, [1, 1, 1], [2, 2, scale], img_flag)
        except Exception as e:
            return None
    
    def resample_image(self, image, spacing, new_spacing, im_flag):
        assert (im_flag=='i' or im_flag=='m')
        shape = np.array(image.shape)
        newSize = np.round(shape * spacing / new_spacing)
        resize_factor = newSize / shape
        resize_factor = resize_factor.tolist()
         
        if im_flag == 'i':
            newVol = zoom(image, resize_factor, mode='nearest', order=2)
        else:
            newVol = zoom(image, resize_factor, mode='constant', order=0)
        return newVol
    
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
    
    def __getitem__(self, idx):
        
        self.data[idx][5]=self.data[idx][5].astype(np.float32)
        
        return self.data[idx][0], self.data[idx][1], self.data[idx][2], \
            self.data[idx][3], self.data[idx][4], self.data[idx][5]
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    
    loader = DatasetLoader(
        path=[
            '/xx/FirstBatch_straighten', # straighten aorta
            '......',
        ],
        save_path='xx/xx',
        threshold={
          'ct': [-100, 300],
          'cta': [-100, 800]
        },
        train_ratio=0.8,
        trans_dict_path='xx/trans_dict.json',
        clip_path='xx/Pretrain/.cache/clip-vit-large-patch14',
        save_verbose=True,
        roi=False,
        discard=10,
        text_mode='all',
        depth=256,
        resolution=[256, 256]
    )
