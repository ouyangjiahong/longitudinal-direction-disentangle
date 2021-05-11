import os
import time
import pdb
from glob import glob
import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.misc as sci
import scipy.ndimage
import pickle
import shutil
import skimage
import skimage.io
import skimage.transform
import skimage.color
from skimage.measure import compare_nrmse, compare_psnr, compare_ssim
import skimage.metrics
import sklearn.metrics
import matplotlib as mpl
import nibabel as nib
import h5py
import pandas as pd
import yaml
import copy

# define dataloader
class CrossSectionalDataset(Dataset):
    def __init__(self, dataset_name, data_img, data_noimg, subj_id_list, case_id_list, aug=False):
        self.data_img = data_img
        self.data_noimg = data_noimg
        self.subj_id_list = subj_id_list
        self.case_id_list = case_id_list
        self.aug = aug

    def __len__(self):
        return len(self.subj_id_list)*3
        # return len(self.subj_id_list)

    def __getitem__(self, idx):
        idx = idx // 3
        subj_id = self.subj_id_list[idx]
        case_id = self.case_id_list[0][idx]
        case_order = self.case_id_list[1][idx]

        if self.aug:
            rand_idx = np.random.randint(0, 10)
        else:
            rand_idx = 0
        img = np.array(self.data_img[subj_id][case_id][rand_idx])

        # img = np.array(self.data_img[subj_id][case_id])
        label = np.array(self.data_noimg[subj_id]['label'])
        age = np.array(self.data_noimg[subj_id]['age'] + self.data_noimg[subj_id]['date_interval'][case_order])
        age = (age - 47.3) / 17.6
        return {'img': img, 'label': label, 'age': age}

class LongitudinalPairDataset(Dataset):
    def __init__(self, dataset_name, data_img, data_noimg, subj_id_list, case_id_list, aug=False):
        self.data_img = data_img
        self.data_noimg = data_noimg
        self.subj_id_list = subj_id_list
        self.case_id_list = case_id_list
        self.aug = aug

    def __len__(self):
        return len(self.subj_id_list)

    def __getitem__(self, idx):
        subj_id = self.subj_id_list[idx]
        case_id_1 = self.case_id_list[0][idx]
        case_id_2 = self.case_id_list[1][idx]
        case_order_1 = self.case_id_list[2][idx]
        case_order_2 = self.case_id_list[3][idx]
        label = np.array(self.data_noimg[subj_id]['label'])
        # label_all = np.array(self.data_noimg[subj_id]['label_all'])[[case_order_1, case_order_2]]
        interval = np.array(self.data_noimg[subj_id]['date_interval'][case_order_2] - self.data_noimg[subj_id]['date_interval'][case_order_1])
        age = np.array(self.data_noimg[subj_id]['age'] + self.data_noimg[subj_id]['date_interval'][case_order_1])
        # print(subj_id, case_order_1, case_order_2, label_all, interval)

        if self.aug:
            rand_idx = np.random.randint(0, 10)
        else:
            rand_idx = 0
        img1 = np.array(self.data_img[subj_id][case_id_1][rand_idx])
        img2 = np.array(self.data_img[subj_id][case_id_2][rand_idx])

        return {'img1': img1, 'img2': img2, 'label': label, 'interval': interval, 'age': age}

class LongitudinalData(object):
    def __init__(self, dataset_name, data_path, img_file_name='ADNI_longitudinal_img.h5',
                noimg_file_name='ADNI_longitudinal_noimg.h5', subj_list_postfix='NC_AD', data_type='single',
                aug=False, batch_size=16, num_fold=5, fold=0, shuffle=True, num_workers=0):
        if dataset_name == 'ADNI' or dataset_name == 'LAB':
            data_img = h5py.File(os.path.join(data_path, img_file_name), 'r')
            data_noimg = h5py.File(os.path.join(data_path, noimg_file_name), 'r')

            subj_id_list_train, case_id_list_train = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_train_'+subj_list_postfix+'.txt'), data_type)
            subj_id_list_val, case_id_list_val = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_val_'+subj_list_postfix+'.txt'), data_type)
            subj_id_list_test, case_id_list_test = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_test_'+subj_list_postfix+'.txt'), data_type)

            if data_type == 'single':
                train_dataset = CrossSectionalDataset(dataset_name, data_img, data_noimg, subj_id_list_train, case_id_list_train, aug=aug)
                val_dataset = CrossSectionalDataset(dataset_name, data_img, data_noimg, subj_id_list_val, case_id_list_val, aug=False)
                test_dataset = CrossSectionalDataset(dataset_name, data_img, data_noimg, subj_id_list_test, case_id_list_test, aug=False)
            elif data_type == 'pair':
                train_dataset = LongitudinalPairDataset(dataset_name, data_img, data_noimg, subj_id_list_train, case_id_list_train, aug=aug)
                val_dataset = LongitudinalPairDataset(dataset_name, data_img, data_noimg, subj_id_list_val, case_id_list_val, aug=False)
                test_dataset = LongitudinalPairDataset(dataset_name, data_img, data_noimg, subj_id_list_test, case_id_list_test, aug=False)
            else:
                raise ValueError('Did not support pair or sequential data yet')

        else:
            raise ValueError('Not support this dataset!')

        self.trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.valLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    def load_idx_list(self, file_path, data_type):
        lines = pd.read_csv(file_path, sep=" ", header=None)
        if data_type == 'single':
            return np.array(lines.iloc[:,0]), [np.array(lines.iloc[:,1]), np.array(lines.iloc[:,2])]
        elif data_type == 'pair':
            return np.array(lines.iloc[:,0]), [np.array(lines.iloc[:,1]),np.array(lines.iloc[:,2]),np.array(lines.iloc[:,3]),np.array(lines.iloc[:,4])]
        else:
            raise ValueError('Not support sequential data type')


# load config file from ckpt
def load_config_yaml(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return True, config
    else:
        return False, None

# save config file at the beginning of the training
def save_config_yaml(ckpt_path, config):
    yaml_path = os.path.join(ckpt_path, 'config.yaml')
    remove_key = []
    for key in config.keys():
        if isinstance(config[key], int) or isinstance(config[key], float) or isinstance(config[key], str) or isinstance(config[key], list)  or isinstance(config[key], dict):
            continue
        remove_key.append(key)
    config_copy = copy.deepcopy(config)
    for key in remove_key:
        config_copy.pop(key, None)
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(config_copy, file)
    print('Saved yaml file')

# load model/scheduler
def load_checkpoint_by_key(values, checkpoint_dir, keys, device, ckpt_name='model_best.pth.tar'):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = os.path.join(checkpoint_dir, ckpt_name)
    print(filename)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            try:
                if key == 'model':
                    values[i] = load_checkpoint_model(values[i], checkpoint[key])
                else:
                    values[i].load_state_dict(checkpoint[key])
                print('loading ' + key + ' success!')
            except:
                print('loading ' + key + ' failed!')
        print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(filename, \
                epoch, checkpoint['monitor_metric']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch

# load each part of the model
def load_checkpoint_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict_filter = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict_filter)
    # load LSSL to LDD
    if 'direction.weight' in pretrained_dict.keys() and 'aging_direction.weight' in model_dict.keys():
        model_dict['aging_direction.weight'] = pretrained_dict['direction.weight']
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

# save results statistics
def save_result_stat(stat, config, info='Default'):
    stat_path = os.path.join(config['ckpt_path'], 'stat.csv')
    columns=['info',] + sorted(stat.keys())
    if not os.path.exists(stat_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(stat_path, mode='a', header=True)

    stat['info'] = info
    for key, value in stat.items():
        stat[key] = [value]
    df = pd.DataFrame.from_dict(stat)
    df = df[columns]
    df.to_csv(stat_path, mode='a', header=False)

def save_checkpoint(state, is_best, checkpoint_dir):
    print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')
        print('save best')

def compute_classification_metrics(label, pred, postfix='NC_AD'):
    if postfix == 'C_single':
        # pdb.set_trace()
        r2 = sklearn.metrics.r2_score(label, pred)
        label = label * 17.6 + 47.3
        pred = pred * 17.6 + 47.3
        mse = sklearn.metrics.mean_squared_error(label, pred, squared=False)
        mae = np.abs(pred - label).mean()
        print(mse, r2, mae)
        return r2
    else:
        pred_bi = (pred>0.5).squeeze(1)
        if 'NC_AD' in postfix:
            classes = [0,2]
        elif 'pMCI_sMCI' in postfix:
            classes = [3,4]
        elif 'C_E_HE' in postfix:
            label = (label > 0)
            classes = [0,1]
        tp = np.sum(np.logical_and(label==classes[1], pred_bi==1))
        fp = np.sum(np.logical_and(label==classes[0], pred_bi==1))
        tn = np.sum(np.logical_and(label==classes[0], pred_bi==0))
        fn = np.sum(np.logical_and(label==classes[1], pred_bi==0))
        auc = sklearn.metrics.roc_auc_score(label==classes[1], pred.squeeze(1))
        sen = tp/(tp+fn)
        spe = tn/(tn+fp)
        bacc = 0.5 * (sen + spe)
        print(auc, bacc, sen, spe)
        return bacc

def compute_average_brain_one_disease(path, model, da, dd, z_list, label_list, age_list, age_thres=[60,85], age_interval=5):
    idx_list = np.logical_and(age_list>age_thres[0], age_list<age_thres[1])
    z_list = z_list[idx_list]
    label_list = label_list[idx_list]
    age_list = age_list[idx_list]
    label_class = np.sort(np.unique(label_list))
    da_norm = da / np.linalg.norm(da)
    dd_norm = dd / np.linalg.norm(dd)
    proj_da_val = np.dot(z_list, np.transpose(da_norm))
    proj_dd_val = np.dot(z_list, np.transpose(dd_norm))

    z_mean_list = []
    for label in label_class:
        # filter this diagnosis cases
        z_list_cls = z_list[label_list==label]
        age_list_cls = age_list[label_list==label]
        # range of age and projection value on aging direction
        age_min, age_max = age_list_cls.min(), age_list_cls.max()
        proj_da_val_cls = proj_da_val[label_list==label]
        da_min, da_max = np.percentile(proj_da_val_cls, 5), np.percentile(proj_da_val_cls, 95)
        if label == 0:
            proj_other_cls = z_list_cls - proj_da_val_cls * np.tile(da_norm, [proj_da_val_cls.shape[0],1])
            proj_other_cls_mean = np.mean(proj_other_cls, 0)
            proj_dd_cls_mean = 0
        else:
            # mean on disease direction
            proj_dd_val_cls = proj_dd_val[label_list==label]
            proj_dd_cls_mean =  np.mean(proj_dd_val_cls, 0) * dd_norm
            proj_other_cls = z_list_cls - np.tile(proj_dd_cls_mean,[proj_da_val_cls.shape[0],1]) - proj_da_val_cls * da_norm
            proj_other_cls_mean = np.transpose(np.mean(proj_other_cls, 0))

        z_mean_age_list = []
        for age in range(age_thres[0], age_thres[1]+1, age_interval):
            da_age = (age - age_thres[0]) / (age_thres[1] - age_thres[0]) * (da_max - da_min) + da_min
            z_mean = da_age * da_norm + proj_other_cls_mean + proj_dd_cls_mean
            z_mean_age_list.append(z_mean)
        z_mean_list.append(np.concatenate(z_mean_age_list, 0))
    z_mean_list = np.stack(z_mean_list, 0)
    recons = model.decoder(torch.Tensor(z_mean_list).to(model.gpu).view(-1, z_mean_list.shape[-1]))
    recons = recons.view(len(label_class), -1, 64, 64, 64).detach().cpu().numpy()
    np.save(path, recons)

def compute_average_brain_two_disease(path, model, da, dd, z_list, label_list, age_list, label_cls, age_thres=[60,85], age_interval=5):
    idx_list = np.logical_and(age_list>age_thres[0], age_list<age_thres[1])
    z_list = z_list[idx_list]
    label_list = label_list[idx_list]
    age_list = age_list[idx_list]
    label_class = np.sort(np.unique(label_list))
    da_norm = da / np.linalg.norm(da)
    dd1, dd2 = dd[:,:da.shape[1]], dd[:,da.shape[1]:]
    dd1_norm = dd1 / np.linalg.norm(dd1)
    dd2_norm = dd2 / np.linalg.norm(dd2)
    proj_da_val = np.dot(z_list, np.transpose(da_norm))
    proj_dd1_val = np.dot(z_list, np.transpose(dd1_norm))
    proj_dd2_val = np.dot(z_list, np.transpose(dd2_norm))

    z_mean_list = []
    for label in label_class:
        # filter this diagnosis cases
        z_list_cls = z_list[label_list==label]
        age_list_cls = age_list[label_list==label]
        # range of age and projection value on aging direction
        age_min, age_max = age_list_cls.min(), age_list_cls.max()
        proj_da_val_cls = proj_da_val[label_list==label]
        da_min, da_max = np.percentile(proj_da_val_cls, 5), np.percentile(proj_da_val_cls, 95)
        if label in label_cls[0]:   # C
            proj_other_cls = z_list_cls - proj_da_val_cls * np.tile(da_norm, [proj_da_val_cls.shape[0],1])
            proj_other_cls_mean = np.mean(proj_other_cls, 0)
            proj_dd1_cls_mean = 0
            proj_dd2_cls_mean = 0
        elif label in label_cls[1] and label in label_cls[2]:   # HE
            # mean on disease direction
            proj_dd1_val_cls = proj_dd1_val[label_list==label]
            proj_dd1_cls_mean =  np.mean(proj_dd1_val_cls, 0) * dd1_norm
            proj_dd2_val_cls = proj_dd2_val[label_list==label]
            proj_dd2_cls_mean =  np.mean(proj_dd2_val_cls, 0) * dd2_norm
            proj_other_cls = z_list_cls - np.tile(proj_dd1_cls_mean,[proj_da_val_cls.shape[0],1]) - np.tile(proj_dd2_cls_mean,[proj_da_val_cls.shape[0],1]) - proj_da_val_cls * da_norm
            proj_other_cls_mean = np.transpose(np.mean(proj_other_cls, 0))
        else:
            if label in label_cls[1]:
                proj_dd1_val_cls = proj_dd1_val[label_list==label]
                proj_dd1_cls_mean =  np.mean(proj_dd1_val_cls, 0) * dd1_norm
                proj_other_cls = z_list_cls - np.tile(proj_dd1_cls_mean,[proj_da_val_cls.shape[0],1]) - proj_da_val_cls * da_norm
                proj_other_cls_mean = np.transpose(np.mean(proj_other_cls, 0))
                proj_dd2_cls_mean = 0
            else:
                proj_dd2_val_cls = proj_dd2_val[label_list==label]
                proj_dd2_cls_mean =  np.mean(proj_dd2_val_cls, 0) * dd2_norm
                proj_other_cls = z_list_cls - np.tile(proj_dd2_cls_mean,[proj_da_val_cls.shape[0],1]) - proj_da_val_cls * da_norm
                proj_other_cls_mean = np.transpose(np.mean(proj_other_cls, 0))
                proj_dd1_cls_mean = 0

        z_mean_age_list = []
        for age in range(age_thres[0], age_thres[1]+1, age_interval):
            da_age = (age - age_thres[0]) / (age_thres[1] - age_thres[0]) * (da_max - da_min) + da_min
            z_mean = da_age * da_norm + proj_other_cls_mean + proj_dd1_cls_mean + proj_dd2_cls_mean
            z_mean_age_list.append(z_mean)
        z_mean_list.append(np.concatenate(z_mean_age_list, 0))
    z_mean_list = np.stack(z_mean_list, 0)
    recons = model.decoder(torch.Tensor(z_mean_list).to(model.gpu).view(-1, z_mean_list.shape[-1]))
    recons = recons.view(len(label_class), -1, 64, 64, 64).detach().cpu().numpy()
    np.save(path, recons)
