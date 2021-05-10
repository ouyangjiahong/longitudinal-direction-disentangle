import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb


class EncoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(EncoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(DecoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class Encoder_var(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
        super(Encoder_var, self).__init__()

        self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.mu = nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1)
        self.log_var = nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        mu = self.mu(conv4)
        log_var = self.log_var(conv4)
        # (16,4,4,4)
        return mu.view(x.shape[0], -1), log_var.view(x.shape[0], -1)

class Encoder(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, dropout=False):
        super(Encoder, self).__init__()

        if dropout:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.1, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.2, num_conv=num_conv)
            self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        else:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # (16,4,4,4)
        return conv4

class Encoder_Var(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
        super(Encoder_Var, self).__init__()

        self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv4_mean = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv4_logvar = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        mean = self.conv4_mean(conv3)
        logvar = self.conv4_logvar(conv3)
        # (16,4,4,4)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, out_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
        super(Decoder, self).__init__()

        self.conv4 = DecoderBlock(inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv3 = DecoderBlock(4*inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv2 = DecoderBlock(2*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv1 = DecoderBlock(inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x_reshaped = x.view(x.shape[0], 16, 4, 4, 4)
        conv4 = self.conv4(x_reshaped)
        conv3 = self.conv3(conv4)
        conv2 = self.conv2(conv3)
        conv1 = self.conv1(conv2)
        output = self.conv0(conv1)
        return output

class Classifier(nn.Module):
    def __init__(self, latent_size=1024, inter_num_ch=64):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
                        # nn.BatchNorm1d(latent_size),
                        # nn.Dropout(0.5),
                        nn.Dropout(0.2),
                        nn.Linear(latent_size, inter_num_ch),
                        nn.LeakyReLU(0.2),
                        # nn.Tanh(),
                        nn.Linear(inter_num_ch, 1))
        self._init()

    def _init(self):
        for layer in self.fc.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.fc(x)


class CLS(nn.Module):
    def __init__(self, latent_size=1024, use_feature=['z', 'delta_z'], dropout=False, gpu=None):
        super(CLS, self).__init__()
        self.gpu = gpu
        self.use_feature = use_feature
        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1, dropout=dropout)
        self.classifier = Classifier(latent_size=len(use_feature)*latent_size, inter_num_ch=64)
        # self.classifier = Classifier(latent_size=len(use_feature)*latent_size, inter_num_ch=16)

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]

        if len(self.use_feature) == 2:
            input = torch.cat([z1, delta_z], 1)
        elif 'z' in self.use_feature:
            input = z1
        else:
            input = delta_z
        pred = self.classifier(input)
        return pred

    def forward_single(self, img1):
        z1 = self.encoder(img1)
        z1 = z1.view(img1.shape[0], -1)
        pred = self.classifier(z1)
        return pred

    def compute_classification_loss(self, pred, label, pos_weight=torch.tensor([2.]), postfix='NC_AD'):
        if postfix == 'C_single':
            loss = nn.MSELoss()(pred.squeeze(1), label)
            return loss, pred
        else:
            if  'NC_AD' in postfix:
                label = label / 2
            elif 'pMCI_sMCI' in postfix:
                label = label - 3
            elif 'C_E_HE' in postfix:
                label = (label > 0).double()
            else:
                raise ValueError('Not support!')
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.gpu, dtype=torch.float))(pred.squeeze(1), label)
            return loss, F.sigmoid(pred)

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder_Var(in_num_ch=1, inter_num_ch=16, num_conv=1)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs_mu, zs_logvar = self.encoder(torch.cat([img1, img2], 0))
        zs = self._sample(zs_mu, zs_logvar)
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2, zs_mu.view(bs*2, -1), zs_logvar.view(bs*2, -1)], [recon1, recon2]

    def _sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # kl loss
    def compute_kl_loss(self, mu, logvar):
        # pdb.set_trace()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return torch.mean(torch.sum(kl, dim=-1))

class LSSL(nn.Module):
    def __init__(self, gpu='None'):
        super(LSSL, self).__init__()
        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)
        self.direction = nn.Linear(1, 1024)
        self.gpu = gpu

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # direction loss
    def compute_direction_loss(self, zs):
        z1, z2 = zs[0], zs[1]
        bs = z1.shape[0]
        delta_z = z2 - z1
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        d_vec = self.direction(torch.ones(bs, 1).to(self.gpu))
        d_vec_norm = torch.norm(d_vec, dim=1) + 1e-12
        cos = torch.sum(delta_z * d_vec, 1) / (delta_z_norm * d_vec_norm)
        return (1. - cos).mean()

class LSP(nn.Module):
    def __init__(self, model_name='LSP', latent_size=1024, num_neighbours=3, agg_method='gaussain', gpu=None):
        super(LSP, self).__init__()
        self.model_name = model_name
        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)
        if latent_size < 1024:
            self.mapping = nn.Linear(1024, latent_size)
        else:
            self.mapping = nn.Sequential()
        self.num_neighbours = num_neighbours
        self.agg_method = agg_method
        self.gpu = gpu

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = self.mapping(zs.view(bs*2, -1))
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    def build_graph_batch(self, zs):
        z1 = zs[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, bs).to(self.gpu)
        for i in range(bs):
            for j in range(i+1, bs):
                dis_mx[i, j] = torch.sum((z1[i] - z1[j]) ** 2)
                dis_mx[j, i] = dis_mx[i, j]
        if self.agg_method == 'gaussian':
            adj_mx = torch.exp(-dis_mx/100)
        if self.num_neighbours < bs:
            adj_mx_filter = torch.zeros(bs, bs).to(self.gpu)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_neighbours+1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
                adj_mx_filter[i, i] = 0.
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.gpu))

    def build_graph_dataset(self, zs_all, zs):
        z1_all = zs_all[0]
        z1 = zs[0]
        ds = z1_all.shape[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, ds).to(self.gpu)
        for i in range(bs):
            for j in range(ds):
                dis_mx[i, j] = torch.sum((z1[i] - z1_all[j]) ** 2)
        if self.agg_method == 'gaussian':
            adj_mx = torch.exp(-dis_mx/100)
        if self.num_neighbours < bs:
            adj_mx_filter = torch.zeros(bs, ds).to(self.gpu)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_neighbours+1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.gpu))

    def compute_social_pooling_delta_z_batch(self, zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z) / adj_mx.sum(1, keepdim=True)    # [bs, ls]
        return delta_z, delta_h

    def compute_social_pooling_delta_z_dataset(self, zs_all, interval_all, zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]
        z1_all, z2_all = zs_all[0], zs_all[1]
        delta_z_all = (z2_all - z1_all) / interval_all.unsqueeze(1)      # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z_all) / adj_mx.sum(1, keepdim=True)    # [bs, ls]
        return delta_z, delta_h

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # direction loss, 1 - cos<delta_z, delta_h>
    def compute_direction_loss(self, delta_z, delta_h):
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        delta_h_norm = torch.norm(delta_h, dim=1) + 1e-12
        cos = torch.sum(delta_z * delta_h, 1) / (delta_z_norm * delta_h_norm)
        return (1. - cos).mean()

    # distance loss, (delta_h - delta_z) / norm_delta_z
    def compute_distance_loss(self, delta_z, delta_h):
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        dis = torch.norm(delta_z - delta_h, dim=1)
        return (dis / delta_z_norm).mean()


class LDD(nn.Module):
    def __init__(self, gpu='None'):
        super(LDD, self).__init__()
        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)
        self.aging_direction = nn.Linear(1, 1024, bias=False)
        self.disease_direction = nn.Linear(1, 1024-1, bias=False)
        self.gpu = gpu

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # direction loss
    def compute_direction_loss(self, zs, labels, interval):
        z1, z2 = zs[0], zs[1]
        bs = z1.shape[0]
        delta_z = (z2 - z1) / interval.unsqueeze(1)
        labels_dis = (labels > 0).type(torch.FloatTensor).to(self.gpu)
        labels_nc = (labels == 0).type(torch.FloatTensor).to(self.gpu)

        num_nc = labels_nc.sum() + 1e-12
        num_dis = labels_dis.sum() + 1e-12

        # define directions
        da_vec = self.aging_direction(torch.ones(bs, 1).to(self.gpu))   # (1024,)
        dd_vec_front = self.disease_direction(torch.ones(bs, 1).to(self.gpu))   # (1023,)
        dd_vec_last = - torch.sum(da_vec[:,:-1] * dd_vec_front, 1) / (da_vec[:,-1] + 1e-12)
        dd_vec = torch.cat([dd_vec_front, dd_vec_last.unsqueeze(1)], 1)
        if torch.any((torch.sum(da_vec * dd_vec, 1) - 1e-4) > 0):
            pdb.set_trace()

        # aging direction
        da_vec_norm = torch.norm(da_vec, dim=1) + 1e-12
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        cos_da = torch.sum(delta_z * da_vec, 1) / (delta_z_norm * da_vec_norm)
        loss_da = (1 - cos_da) * labels_nc
        loss_da = loss_da.sum() / num_nc

        # disease direction
        dd_vec_norm = torch.norm(dd_vec, dim=1) + 1e-12
        delta_z_da_proj = torch.sum(delta_z * da_vec, 1) / da_vec_norm
        delta_z_da = delta_z_da_proj.unsqueeze(1) * da_vec / da_vec_norm.unsqueeze(1)
        delta_z_dd = delta_z - delta_z_da
        delta_z_dd_norm = torch.norm(delta_z_dd, dim=1) + 1e-12
        cos_dd = torch.sum(delta_z_dd * dd_vec, 1) / (delta_z_dd_norm * dd_vec_norm)
        loss_dd = (1 - cos_dd) * labels_dis
        loss_dd = loss_dd.sum() / num_dis


        # kl loss
        nc_idx = torch.where(labels_nc == 1)[0]
        dis_idx = torch.where(labels_dis == 1)[0]
        try:
            delta_z_da_proj_nc = torch.index_select(delta_z_da_proj, 0, nc_idx)
            delta_z_da_proj_dis = torch.index_select(delta_z_da_proj, 0, dis_idx)
            nc_mean = torch.mean(delta_z_da_proj_nc)
            dis_mean = torch.mean(delta_z_da_proj_dis)
            nc_std = torch.std(delta_z_da_proj_nc)
            dis_std = torch.std(delta_z_da_proj_dis)
            # kl_loss = 0.5 * (nc_mean - dis_mean)**2
            if not torch.isfinite(dis_std):
                kl_loss = torch.tensor(0., device=self.gpu)
            else:
                kl_loss = torch.log(dis_std / nc_std) + (nc_std**2 + (nc_mean - dis_mean)**2) / (2*dis_std**2) - 0.5
        except:
            kl_loss = torch.tensor(0., device=self.gpu)


        # penalty loss for NC projection on disease direction
        dis_idx = torch.where(labels_dis == 1)[0]
        delta_z_dd_proj = torch.sum(delta_z * dd_vec, 1) / dd_vec_norm
        delta_z_dd_proj_nc = torch.index_select(delta_z_dd_proj, 0, nc_idx)
        delta_z_dd_proj_dis = torch.index_select(delta_z_dd_proj, 0, dis_idx)
        try:
            # penalty_loss = torch.mean(torch.abs(delta_z_dd_proj_nc / (delta_z_da_proj_nc+1e-12)))
            penalty_loss = torch.mean(torch.abs(delta_z_dd_proj_nc)) / (torch.mean(torch.abs(delta_z_dd_proj_dis))+1e-12)
        except:
            penalty_loss = torch.tensor(0., device=self.gpu)

        return loss_da, loss_dd, kl_loss, penalty_loss

    def compute_directions(self):
        da_vec = self.aging_direction(torch.ones(1, 1).to(self.gpu))   # (1024,)
        dd_vec_front = self.disease_direction(torch.ones(1, 1).to(self.gpu))   # (1023,)
        dd_vec_last = - torch.sum(da_vec[:,:-1] * dd_vec_front, 1) / (da_vec[:,-1] + 1e-12)
        dd_vec = torch.cat([dd_vec_front, dd_vec_last.unsqueeze(1)], 1)
        return da_vec, dd_vec
