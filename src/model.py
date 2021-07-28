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


# class Encoder_var(nn.Module):
#     def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
#         super(Encoder_var, self).__init__()
#
#         self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
#         self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
#         self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
#         self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
#         self.mu = nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1)
#         self.log_var = nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1)
#
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         conv4 = self.conv4(conv3)
#         mu = self.mu(conv4)
#         log_var = self.log_var(conv4)
#         # (16,4,4,4)
#         return mu.view(x.shape[0], -1), log_var.view(x.shape[0], -1)

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

class Encoder_Simple(nn.Module):
    def __init__(self, inter_num_ch=16):
        super(Encoder_Simple, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(1, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2),
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2),
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2),
                        nn.Conv3d(4*inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2))

    def forward(self, x):
        return self.conv(x)

class Encoder_Simple_Mapping(nn.Module):
    def __init__(self, inter_num_ch=16, latent_size=512):
        super(Encoder_Simple_Mapping, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(1, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2),
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2),
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2),
                        nn.Conv3d(4*inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(2))

        self.mapping = nn.Sequential(
                            nn.Linear(1024, latent_size),
                            nn.Tanh())

    def forward(self, x):
        z = self.conv(x)
        z_flatten = z.view(x.shape[0], -1)
        z_flatten_map = self.mapping(z_flatten)
        return z_flatten_map

class Decoder(nn.Module):
    def __init__(self, out_num_ch=1, img_size=(64,64,64), inter_num_ch=16, in_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
        super(Decoder, self).__init__()

        self.conv4 = DecoderBlock(in_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv3 = DecoderBlock(4*inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv2 = DecoderBlock(2*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv1 = DecoderBlock(inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)

    def forward(self, x):
        if len(x.shape) < 5:
            x_reshaped = x.view(x.shape[0], 16, 4, 4, 4)
        else:
            x_reshaped = x
        conv4 = self.conv4(x_reshaped)
        conv3 = self.conv3(conv4)
        conv2 = self.conv2(conv3)
        conv1 = self.conv1(conv2)
        output = self.conv0(conv1)
        return output

class Decoder_Simple(nn.Module):
    def __init__(self, inter_num_ch=16):
        super(Decoder_Simple, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(2*inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(inter_num_ch, 1, kernel_size=3, padding=1))

    def forward(self, x):
        x_reshaped = x.view(x.shape[0], 16, 4, 4, 4)
        return self.conv(x_reshaped)
        # return self.conv(x)

class Decoder_Simple_Mapping(nn.Module):
    def __init__(self, inter_num_ch=16, latent_size=512):
        super(Decoder_Simple_Mapping, self).__init__()
        self.mapping = nn.Sequential(
                            nn.Linear(latent_size, 1024),
                            nn.Tanh())

        self.conv = nn.Sequential(
                        nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(2*inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
                        nn.Conv3d(inter_num_ch, 1, kernel_size=3, padding=1))

    def forward(self, x):
        x_map = self.mapping(x)
        x_reshaped = x_map.view(x.shape[0], 16, 4, 4, 4)
        return self.conv(x_reshaped)
        # return self.conv(x)

class Classifier(nn.Module):
    def __init__(self, latent_size=1024, inter_num_ch=64):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
                        # nn.BatchNorm1d(latent_size),
                        # nn.Dropout(0.5),
                        # nn.Dropout(0.2),
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

    # def compute_classification_loss(self, pred, label, pos_weight=torch.tensor([2.]), postfix='NC_AD'):
    #     if postfix == 'C_single':
    #         loss = nn.MSELoss()(pred.squeeze(1), label)
    #         return loss, pred
    #     else:
    #         if  'NC_AD' in postfix:
    #             label = label / 2
    #         elif 'pMCI_sMCI' in postfix:
    #             label = label - 3
    #         elif 'C_E_HE' in postfix:
    #             label = (label > 0).double()
    #         else:
    #             raise ValueError('Not support!')
    #         loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.gpu, dtype=torch.float))(pred.squeeze(1), label)
    #         return loss, F.sigmoid(pred)

    def compute_loss(self, pred, label, downstream='sMCI_pMCI', pos_weight=None):
        # pdb.set_trace()
        if downstream == 'adas' or downstream == 'future_adas':
            loss = nn.MSELoss()(pred.squeeze(1), label)
        elif downstream == 'sMCI_pMCI' or downstream == 'amyloid':
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.gpu, dtype=torch.float))(pred.squeeze(1), label)
            pred = F.sigmoid(pred)
        else:
            raise ValueError('Not support!')
        return loss, pred

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

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = Encoder_Var(in_num_ch=2, inter_num_ch=16, num_conv=1)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, in_num_ch=17, num_conv=1)

    def forward(self, img1, img2, interval, label):
        bs = img1.shape[0]
        input1 = torch.cat([img1, label.reshape(-1,1,1,1,1).repeat(1,1,64,64,64)], 1)
        input2 = torch.cat([img2, label.reshape(-1,1,1,1,1).repeat(1,1,64,64,64)], 1)
        zs_mu, zs_logvar = self.encoder(torch.cat([input1, input2], 0))
        zs = self._sample(zs_mu, zs_logvar)
        # zs = zs.reshape(bs*2, 16, 4, 4, 4)
        zs_cond = torch.cat([zs, label.reshape(-1,1,1,1,1).repeat(2,1,4,4,4)], 1)
        recons = self.decoder(zs_cond)
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
    def __init__(self, gpu='None', model='normal', is_mapping=False, latent_size=512):
        super(LSSL, self).__init__()
        if model == 'normal':
            self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
            self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)
        elif model == 'simple':
            self.encoder = Encoder_Simple()
            self.decoder = Decoder_Simple()
        else:
            raise ValueError('Do not support other encoder-decoder model!')
        self.is_mapping = is_mapping
        if self.is_mapping:
            self.mapping_enc = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(1024, latent_size),
                                nn.Tanh())
            self.mapping_dec = nn.Sequential(
                                nn.Linear(latent_size, 1024),
                                nn.Tanh())
        self.direction = nn.Linear(1, latent_size)
        self.gpu = gpu

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        if self.is_mapping:
            zs = self.mapping_enc(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]

        if self.is_mapping:
            zs = self.mapping_dec(zs)
        recons = self.decoder(zs)

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

    def compute_directions(self):
        # pdb.set_trace()
        da_vec = self.direction(torch.ones(1, 1).to(self.gpu))   # (1024,)
        return da_vec

class wLSSL1(nn.Module):
    def __init__(self, gpu='None', model='normal', is_mapping=False, latent_size=512):
        super(wLSSL1, self).__init__()
        if model == 'normal':
            self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
            self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)
        elif model == 'simple':
            self.encoder = Encoder_Simple()
            self.decoder = Decoder_Simple()
        else:
            raise ValueError('Do not support other encoder-decoder model!')
        self.weak_classifier = Classifier(latent_size=latent_size, inter_num_ch=64)
        self.is_mapping = is_mapping
        if self.is_mapping:
            self.mapping_enc = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(1024, latent_size),
                                nn.Tanh())
            self.mapping_dec = nn.Sequential(
                                nn.Linear(latent_size, 1024),
                                nn.Tanh())
        self.direction = nn.Linear(1, latent_size)
        self.gpu = gpu

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        if self.is_mapping:
            zs = self.mapping_enc(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]

        if self.is_mapping:
            zs = self.mapping_dec(zs)
        recons = self.decoder(zs)

        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    # classification loss
    def compute_classification_loss(self, zs, label):
        pred = self.weak_classifier(torch.cat(zs, 0))
        loss = nn.BCEWithLogitsLoss()(pred.squeeze(1), label.repeat(2))
        return loss

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

    def compute_directions(self):
        # pdb.set_trace()
        da_vec = self.direction(torch.ones(1, 1).to(self.gpu))   # (1024,)
        return da_vec

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
    def __init__(self, gpu='None', model='normal', is_mapping=False, latent_size=512):
        super(LDD, self).__init__()
        if model == 'normal':
            self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
            self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)
        elif model == 'simple':
            self.encoder = Encoder_Simple()
            self.decoder = Decoder_Simple()
        else:
            raise ValueError('Do not support other encoder-decoder model!')
        self.is_mapping = is_mapping
        if self.is_mapping:
            self.mapping_enc = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(1024, latent_size),
                                nn.Tanh())
            self.mapping_dec = nn.Sequential(
                                nn.Linear(latent_size, 1024),
                                nn.Tanh())

        self.aging_direction = nn.Linear(1, latent_size, bias=False)
        self.disease_direction = nn.Linear(1, latent_size-1, bias=False)
        self.gpu = gpu

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        if self.is_mapping:
            zs = self.mapping_enc(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]

        if self.is_mapping:
            zs = self.mapping_dec(zs)
        recons = self.decoder(zs)

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


class LDDM(nn.Module):
    def __init__(self, label_list, gpu='None', model='normal', is_mapping=False, latent_size=512):
        super(LDDM, self).__init__()
        if model == 'normal':
            self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=1)
            self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=1)
        elif model == 'simple':
            self.encoder = Encoder_Simple()
            self.decoder = Decoder_Simple()
        else:
            raise ValueError('Do not support other encoder-decoder model!')
        self.is_mapping = is_mapping
        if self.is_mapping:
            self.mapping_enc = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(1024, latent_size),
                                nn.Tanh())
            self.mapping_dec = nn.Sequential(
                                nn.Linear(latent_size, 1024),
                                nn.Tanh())

        self.aging_direction = nn.Linear(1, latent_size, bias=False)
        self.disease_direction1 = nn.Linear(1, latent_size-1, bias=False)
        self.disease_direction2 = nn.Linear(1, latent_size-1, bias=False)
        self.label_list = label_list            # [[NC labels], [dis1 labels], [dis2 labels]]
        self.gpu = gpu

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        if self.is_mapping:
            zs = self.mapping_enc(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]

        if self.is_mapping:
            zs = self.mapping_dec(zs)
        recons = self.decoder(zs)

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

        # nc
        labels_nc = torch.zeros_like(labels).type(torch.FloatTensor).to(self.gpu)
        for i in range(len(self.label_list[0])):
            labels_nc += labels==self.label_list[0][i]
        labels_nc = labels_nc > 0
        # disease 1 only
        labels_dis1 = torch.zeros_like(labels).type(torch.FloatTensor).to(self.gpu) # subjects only have disease 1
        label_list_dis1_only = np.setdiff1d(self.label_list[1], self.label_list[2])
        for i in label_list_dis1_only:
            labels_dis1 += labels==i
        labels_dis1 = labels_dis1 > 0
        # disease 2 only
        label_list_dis2_only = np.setdiff1d(self.label_list[2], self.label_list[1])
        labels_dis2 = torch.zeros_like(labels).type(torch.FloatTensor).to(self.gpu)
        for i in label_list_dis2_only:
            labels_dis2 += labels==i
        labels_dis2 = labels_dis2 > 0
        # disease 1 & 2
        label_list_dis12 = np.intersect1d(self.label_list[1], self.label_list[2])
        labels_dis12 = torch.zeros_like(labels).type(torch.FloatTensor).to(self.gpu)
        for i in label_list_dis12:
            labels_dis12 += labels==i
        labels_dis12 = labels_dis12 > 0

        num_nc = labels_nc.sum() + 1e-12
        num_dis1 = labels_dis1.sum() + 1e-12
        num_dis2 = labels_dis2.sum() + 1e-12
        num_dis12 = labels_dis12.sum() + 1e-12

        # define directions
        # aging direction
        da_vec = self.aging_direction(torch.ones(bs, 1).to(self.gpu))   # (1024,)
        da_vec_norm = torch.norm(da_vec, dim=1) + 1e-12
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        cos_da = torch.sum(delta_z * da_vec, 1) / (delta_z_norm * da_vec_norm)
        loss_da = (1 - cos_da) * labels_nc
        loss_da = loss_da.sum() / num_nc

        # component on aging direction
        delta_z_da_proj = torch.sum(delta_z * da_vec, 1) / da_vec_norm
        delta_z_da = delta_z_da_proj.unsqueeze(1) * da_vec / da_vec_norm.unsqueeze(1)
        # residual component
        delta_z_dd = delta_z - delta_z_da
        delta_z_dd_norm = torch.norm(delta_z_dd, dim=1) + 1e-12

        # disease direction 1
        dd1_vec_front = self.disease_direction1(torch.ones(bs, 1).to(self.gpu))   # (1023,)
        dd1_vec_last = - torch.sum(da_vec[:,:-1] * dd1_vec_front, 1) / (da_vec[:,-1] + 1e-12)
        dd1_vec = torch.cat([dd1_vec_front, dd1_vec_last.unsqueeze(1)], 1)
        if torch.any((torch.sum(da_vec * dd1_vec, 1) - 1e-4) > 0):
            pdb.set_trace()
        dd1_vec_norm = torch.norm(dd1_vec, dim=1) + 1e-12

        # disease direction 2
        dd2_vec_front = self.disease_direction2(torch.ones(bs, 1).to(self.gpu))   # (1023,)
        dd2_vec_last = - torch.sum(da_vec[:,:-1] * dd2_vec_front, 1) / (da_vec[:,-1] + 1e-12)
        dd2_vec = torch.cat([dd2_vec_front, dd2_vec_last.unsqueeze(1)], 1)
        if torch.any((torch.sum(da_vec * dd2_vec, 1) - 1e-4) > 0):
            pdb.set_trace()
        dd2_vec_norm = torch.norm(dd2_vec, dim=1) + 1e-12

        # disease direction loss
        cos_dd1 = torch.sum(delta_z_dd * dd1_vec, 1) / (delta_z_dd_norm * dd1_vec_norm)
        cos_dd2 = torch.sum(delta_z_dd * dd2_vec, 1) / (delta_z_dd_norm * dd2_vec_norm)

        mode_cosdd = 0
        if mode_cosdd == 1:       # HE compute separately
            loss_dd1 = (1 - cos_dd1) * labels_dis1
            loss_dd1 = loss_dd1.sum() / num_dis1

            loss_dd2 = (1 - cos_dd2) * labels_dis2
            loss_dd2 = loss_dd2.sum() / num_dis2

            # TODO, for group HE, compute
            # loss on d2
            delta_z_dd1_proj = torch.sum(delta_z_dd * dd1_vec, 1) / dd1_vec_norm
            delta_z_dd1 = delta_z_dd1_proj.unsqueeze(1) * dd1_vec / dd1_vec_norm.unsqueeze(1)
            delta_z_dd2_res = delta_z_dd - delta_z_dd1
            delta_z_dd2_res_norm = torch.norm(delta_z_dd2_res, dim=1) + 1e-12
            cos_dd2_12 = torch.sum(delta_z_dd2_res * dd2_vec, 1) / (delta_z_dd2_res_norm * dd2_vec_norm)
            loss_dd2_12 = (1 - cos_dd2_12) * labels_dis12
            loss_dd2_12 = loss_dd2_12.sum() / num_dis12
            # loss on d1
            delta_z_dd2_proj = torch.sum(delta_z_dd * dd2_vec, 1) / dd2_vec_norm
            delta_z_dd2 = delta_z_dd2_proj.unsqueeze(1) * dd2_vec / dd2_vec_norm.unsqueeze(1)
            delta_z_dd1_res = delta_z_dd - delta_z_dd2
            delta_z_dd1_res_norm = torch.norm(delta_z_dd1_res, dim=1) + 1e-12
            cos_dd1_12 = torch.sum(delta_z_dd1_res * dd1_vec, 1) / (delta_z_dd1_res_norm * dd1_vec_norm)
            loss_dd1_12 = (1 - cos_dd1_12) * labels_dis12
            loss_dd1_12 = loss_dd1_12.sum() / num_dis12

            loss_dd = loss_dd1 + loss_dd2 + 0.5*(loss_dd1_12+loss_dd2_12)
        elif mode_cosdd == 2:        # HE together with H/E
            loss_dd1 = (1 - cos_dd1) * (labels_dis1 + labels_dis12)
            loss_dd1 = loss_dd1.sum() / (num_dis1 + num_dis12)

            loss_dd2 = (1 - cos_dd2) * (labels_dis2 + labels_dis12)
            loss_dd2 = loss_dd2.sum() / (num_dis2 + num_dis12)
            loss_dd = loss_dd1 + loss_dd2
        else:       # do not regularize HE
            loss_dd1 = (1 - cos_dd1) * labels_dis1
            loss_dd1 = loss_dd1.sum() / num_dis1

            loss_dd2 = (1 - cos_dd2) * labels_dis2
            loss_dd2 = loss_dd2.sum() / num_dis2
            loss_dd = loss_dd1 + loss_dd2


        # kl loss
        nc_idx = torch.where(labels_nc == 1)[0]
        mode_kl = 1
        if mode_kl == 1:        # do not include HE
            dis1_idx = torch.where(labels_dis1)[0]
            dis2_idx = torch.where(labels_dis2)[0]
        else:
            dis1_idx = torch.where(labels_dis1 | labels_dis12)[0]
            dis2_idx = torch.where(labels_dis2 | labels_dis12)[0]
        try:
            delta_z_da_proj_nc = torch.index_select(delta_z_da_proj, 0, nc_idx)
            delta_z_da_proj_dis1 = torch.index_select(delta_z_da_proj, 0, dis1_idx)
            delta_z_da_proj_dis2 = torch.index_select(delta_z_da_proj, 0, dis2_idx)
            nc_mean = torch.mean(delta_z_da_proj_nc)
            dis1_mean = torch.mean(delta_z_da_proj_dis1)
            dis2_mean = torch.mean(delta_z_da_proj_dis2)
            nc_std = torch.std(delta_z_da_proj_nc)
            dis1_std = torch.std(delta_z_da_proj_dis1)
            dis2_std = torch.std(delta_z_da_proj_dis2)
            # kl_loss = 0.5 * (nc_mean - dis_mean)**2
            kl_loss = torch.tensor(0., device=self.gpu)
            if torch.isfinite(dis1_std):
                kl_loss += torch.log(dis1_std / nc_std) + (nc_std**2 + (nc_mean - dis1_mean)**2) / (2*dis1_std**2) - 0.5
            if torch.isfinite(dis2_std):
                kl_loss += torch.log(dis2_std / nc_std) + (nc_std**2 + (nc_mean - dis2_mean)**2) / (2*dis2_std**2) - 0.5
        except:
            kl_loss = torch.tensor(0., device=self.gpu)


        # penalty loss for NC projection on disease direction
        penalty_loss = torch.tensor(0., device=self.gpu)
        delta_z_dd1_proj = torch.sum(delta_z * dd1_vec, 1) / dd1_vec_norm
        delta_z_dd1_proj_dis = torch.index_select(delta_z_dd1_proj, 0, dis1_idx)

        mode_penalty = False
        if mode_penalty:   # penalty of C/H on dd1
            dis_not1_idx = torch.where(labels_nc | labels_dis2)[0]
            delta_z_dd1_proj_nc = torch.index_select(delta_z_dd1_proj, 0, dis_not1_idx)
        else:      # penalty of C on dd1
            delta_z_dd1_proj_nc = torch.index_select(delta_z_dd1_proj, 0, nc_idx)

        try:
            # penalty_loss = torch.mean(torch.abs(delta_z_dd_proj_nc / (delta_z_da_proj_nc+1e-12)))
            penalty_loss += torch.mean(torch.abs(delta_z_dd1_proj_nc)) / (torch.mean(torch.abs(delta_z_dd1_proj_dis))+1e-12)
        except:
            penalty_loss += torch.tensor(0., device=self.gpu)

        delta_z_dd2_proj = torch.sum(delta_z * dd2_vec, 1) / dd1_vec_norm

        delta_z_dd2_proj_dis = torch.index_select(delta_z_dd2_proj, 0, dis2_idx)
        if mode_penalty:   # penalty of C/E on dd2
            dis_not2_idx = torch.where(labels_nc | labels_dis1)[0]
            delta_z_dd2_proj_nc = torch.index_select(delta_z_dd2_proj, 0, dis_not2_idx)
        else:      # penalty of C on dd2
            delta_z_dd2_proj_nc = torch.index_select(delta_z_dd2_proj, 0, nc_idx)

        try:
            # penalty_loss = torch.mean(torch.abs(delta_z_dd_proj_nc / (delta_z_da_proj_nc+1e-12)))
            penalty_loss += torch.mean(torch.abs(delta_z_dd2_proj_nc)) / (torch.mean(torch.abs(delta_z_dd2_proj_dis))+1e-12)
        except:
            penalty_loss += torch.tensor(0., device=self.gpu)

        return loss_da, loss_dd, kl_loss, penalty_loss

    def compute_directions(self):
        da_vec = self.aging_direction(torch.ones(1, 1).to(self.gpu))   # (1024,)
        dd1_vec_front = self.disease_direction1(torch.ones(1, 1).to(self.gpu))   # (1023,)
        dd1_vec_last = - torch.sum(da_vec[:,:-1] * dd1_vec_front, 1) / (da_vec[:,-1] + 1e-12)
        dd1_vec = torch.cat([dd1_vec_front, dd1_vec_last.unsqueeze(1)], 1)
        dd2_vec_front = self.disease_direction2(torch.ones(1, 1).to(self.gpu))   # (1023,)
        dd2_vec_last = - torch.sum(da_vec[:,:-1] * dd2_vec_front, 1) / (da_vec[:,-1] + 1e-12)
        dd2_vec = torch.cat([dd2_vec_front, dd2_vec_last.unsqueeze(1)], 1)
        return da_vec, torch.cat([dd1_vec, dd2_vec], 1)
