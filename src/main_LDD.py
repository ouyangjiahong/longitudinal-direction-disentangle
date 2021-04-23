import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import tqdm
import psutil
# from torch.utils.tensorboard import SummaryWriter

from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

_, config = load_config_yaml('config.yaml')
config['device'] = torch.device('cuda:'+ config['gpu'])

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)


# ckpt folder, load yaml config
config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
elif config['load_yaml']:       # exist and use yaml config
    print('Load config ', os.path.join(config['ckpt_path'], 'config.yaml'))
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config.yaml'))
    if flag:    # load yaml success
        print('load yaml config file')
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == 'phase' or key == 'gpu' or key == 'continue_train' or key == 'ckpt_name':
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print('current config do not have yaml param')
    else:
        save_config_yaml(config['ckpt_path'], config)
print(config)

# define dataset
Data = LongitudinalData(config['dataset_name'], config['data_path'], img_file_name=config['img_file_name'],
            noimg_file_name=config['noimg_file_name'], subj_list_postfix=config['subj_list_postfix'],
            data_type=config['data_type'], batch_size=config['batch_size'], num_fold=config['num_fold'],
            fold=config['fold'], shuffle=config['shuffle'])
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader

# define model
if config['model_name'] == 'LDD':
    model = LDD(gpu=config['device']).to(config['device'])
elif config['model_name'] == 'AE':
    model = AE().to(config['device'])
elif config['model_name'] == 'VAE':
    model = VAE().to(config['device'])
elif config['model_name'] == 'LSSL':
    model = LSSL(gpu=config['device']).to(config['device'])
elif config['model_name'] in ['LSP']:
    model = LSP(gpu=config['device']).to(config['device'])
else:
    raise ValueError('Not support other models yet!')

# define optimizer
if config['froze_dir_a']:
    try:
        model.aging_direction.requires_grad = False
        print('Frozen aging direction!!!')
    except:
        print('Model does not have aging_direction')

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    # [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    print('starting lr:', optimizer.param_groups[0]['lr'])
else:
    start_epoch = -1

# writer = SummaryWriter()

def train():
    global_iter = 0
    monitor_metric_best = 100
    start_time = time.time()

    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        loss_all_dict = {'all': 0, 'recon': 0., 'dir_a': 0., 'dir_d': 0., 'kl': 0., 'penalty': 0.}
        global_iter0 = global_iter
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            img1 = sample['img1'].to(config['device'], dtype=torch.float).unsqueeze(1)
            img2 = sample['img2'].to(config['device'], dtype=torch.float).unsqueeze(1)
            label = sample['label'].to(config['device'], dtype=torch.float)
            interval = sample['interval'].to(config['device'], dtype=torch.float)

            if img1.shape[0] <= config['batch_size'] // 2:
                break

            # run model
            zs, recons = model(img1, img2, interval)

            # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(img1, recons[0]) + model.compute_recon_loss(img2, recons[1]))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_dir_a'] > 0 or config['lambda_dir_d'] > 0 or config['lambda_kl'] > 0:
                loss_dir_a, loss_dir_d, loss_kl, loss_penalty = model.compute_direction_loss(zs, label, interval)
                if config['lambda_dir_a'] > 0:
                    loss += config['lambda_dir_a'] * loss_dir_a
                if config['lambda_dir_d'] > 0:
                    loss += config['lambda_dir_d'] * loss_dir_d
                if config['lambda_kl'] > 0:
                    loss += config['lambda_kl'] * loss_kl
                if config['lambda_penalty'] > 0:
                    loss += config['lambda_penalty'] * loss_penalty
            else:
                loss_dir_a = torch.tensor(0.)
                loss_dir_d = torch.tensor(0.)
                loss_kl = torch.tensor(0.)
                loss_penalty = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['dir_a'] += loss_dir_a.item()
            loss_all_dict['dir_d'] += loss_dir_d.item()
            loss_all_dict['kl'] += loss_kl.item()
            loss_all_dict['penalty'] += loss_penalty.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            optimizer.zero_grad()

            if global_iter % 1 == 0:
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon=[%.4f], dir_a=[%.4f], dir_d=[%.4f], kl=[%.4f], penalty=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon.item(), loss_dir_a.item(), loss_dir_d.item(), loss_kl.item(), loss_penalty.item()))

            # if iter > 2:
            #     break

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
            # writer.add_scalar("train/"+key, loss_all_dict[key], epoch)
        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['all']
        scheduler.step(monitor_metric)
        save_result_stat(stat, config, info='val')
        # for key in stat.keys():
        #     writer.add_scalar("val/"+key, stat[key], epoch)
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        save_checkpoint(state, is_best, config['ckpt_path'])

def evaluate(phase='val', set='val', save_res=True, info=''):
    model.eval()
    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    res_path = os.path.join(config['ckpt_path'], 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        # raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'recon': 0., 'dir_a': 0., 'dir_d': 0., 'kl': 0., 'penalty': 0.}
    img1_list = []
    img2_list = []
    label_list = []
    recon1_list = []
    recon2_list = []
    z1_list = []
    z2_list = []
    interval_list = []
    age_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            img1 = sample['img1'].to(config['device'], dtype=torch.float).unsqueeze(1)
            img2 = sample['img2'].to(config['device'], dtype=torch.float).unsqueeze(1)
            label = sample['label'].to(config['device'], dtype=torch.float)
            interval = sample['interval'].to(config['device'], dtype=torch.float)

            # run model
            zs, recons = model(img1, img2, interval)

            # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(img1, recons[0]) + model.compute_recon_loss(img2, recons[1]))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_dir_a'] > 0 or config['lambda_dir_d'] > 0 or config['lambda_kl'] > 0:
                loss_dir_a, loss_dir_d, loss_kl, loss_penalty = model.compute_direction_loss(zs, label, interval)
                if config['lambda_dir_a'] > 0:
                    loss += config['lambda_dir_a'] * loss_dir_a
                if config['lambda_dir_d'] > 0:
                    loss += config['lambda_dir_d'] * loss_dir_d
                if config['lambda_kl'] > 0:
                    loss += config['lambda_kl'] * loss_kl
                if config['lambda_penalty'] > 0:
                    loss += config['lambda_penalty'] * loss_penalty
            else:
                loss_dir_a = torch.tensor(0.)
                loss_dir_d = torch.tensor(0.)
                loss_kl = torch.tensor(0.)
                loss_penalty = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['dir_a'] += loss_dir_a.item()
            loss_all_dict['dir_d'] += loss_dir_d.item()
            loss_all_dict['kl'] += loss_kl.item()
            loss_all_dict['penalty'] += loss_kl.item()


            if phase == 'test' and save_res:
                # img1_list.append(img1.detach().cpu().numpy())
                # img2_list.append(img2.detach().cpu().numpy())
                # recon1_list.append(recons[0].detach().cpu().numpy())
                # recon2_list.append(recons[1].detach().cpu().numpy())
                z1_list.append(zs[0].detach().cpu().numpy())
                z2_list.append(zs[1].detach().cpu().numpy())
                interval_list.append(interval.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())
                label_list.append(label.detach().cpu().numpy())

            # if iter > 2:
            #     break

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= iter

        if phase == 'test' and save_res:
            pdb.set_trace()
            # img1_list = np.concatenate(img1_list, axis=0)
            # img2_list = np.concatenate(img2_list, axis=0)
            # recon1_list = np.concatenate(recon1_list, axis=0)
            # recon2_list = np.concatenate(recon2_list, axis=0)
            z1_list = np.concatenate(z1_list, axis=0)
            z2_list = np.concatenate(z2_list, axis=0)
            interval_list = np.concatenate(interval_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            label_list = np.concatenate(label_list, axis=0)
            da, dd = model.compute_directions()
            h5_file = h5py.File(path, 'w')
            # h5_file.create_dataset('img1', data=img1_list)
            # h5_file.create_dataset('img2', data=img2_list)
            h5_file.create_dataset('label', data=label_list)
            # h5_file.create_dataset('recon1', data=recon1_list)
            # h5_file.create_dataset('recon2', data=recon2_list)
            h5_file.create_dataset('z1', data=z1_list)
            h5_file.create_dataset('z2', data=z2_list)
            h5_file.create_dataset('interval', data=interval_list)
            h5_file.create_dataset('age', data=age_list)
            h5_file.create_dataset('da', data=da.detach().cpu().numpy())
            h5_file.create_dataset('dd', data=dd.detach().cpu().numpy())

    return loss_all_dict

if config['phase'] == 'train':
    train()
else:
    stat = evaluate(phase='test', set='train', save_res=True, info='')
    # stat = evaluate(phase='test', set='test', save_res=True, info='')
    print(stat)
