import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import os
import SimpleITK as sitk
import numpy as np
import monai

from dataLoader import Loader
import imageio
import yaml

class Train(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(self.config)
        if not os.path.exists(self.config['MODEL_SAVE_PATH']):
            os.makedirs(self.config['MODEL_SAVE_PATH'])
        self.device = 'cuda:' + str(self.config['GPU1'])

        self.l1loss = torch.nn.L1Loss()
        self.train_loader = DataLoader(Loader(self.config['MODE'], self.config['DIM'], self.config['DEG'], trainMerge=self.config['TRAIN_MERGE']), batch_size=self.config['BATCH_SIZE'], shuffle=True)
        
        if self.config['DIM'] == 3:
            self.model = monai.networks.nets.UNet(3,1,1,(32,64,128),(2,2), num_res_units=4)
        elif self.config['DIM'] == 2:
            self.model = monai.networks.nets.UNet(2,1,1,(64,128,256),(2,2), num_res_units=4)
        
        if self.config['TRAIN_MERGE']:
            self.model = monai.networks.nets.UNet(3,2,1,(32,64,128),(2,2), num_res_units=4)

        self.opt = torch.optim.Adam(self.model.parameters(), self.config['LR'], [self.config['BETA1'], self.config['BETA2']], weight_decay=self.config['WEIGHT_DECAY'])

        self.set_gpu()

    def set_gpu(self):
        self.model = self.model.to(self.device)

    def model_save(self, iteration):
        print('saving')
        self.model = self.model.cpu()
        
        torch.save(self.model.state_dict(),
            os.path.join(self.config['MODEL_SAVE_PATH'], 'unet%dd_%s_%d.pth' % (self.config['DIM'], self.config['SAVE_NAME'], iteration)))
        
        self.set_gpu()

    def train3d(self):
        def norm(x):
            return (x-np.min(x))/(np.max(x)-np.min(x))

        for epoch in range(self.config['NUM_EPOCH']):
            for i, data in enumerate(self.train_loader):
                numIter = epoch*len(self.train_loader)+i
                # print('epoch', epoch, 'num:', i, 'iter:', numIter)
                volume_fbp_rs = data['volume_fbp_rs'].to(self.device)
                volume_fbp = data['volume_fbp'].to(self.device)
                # print(volume_fbp_rs.shape, volume_fbp_rs.shape)

                fbp_pred = self.model(volume_fbp)
                print(fbp_pred.max().item(), fbp_pred.min().item())
                loss = 25*self.l1loss(fbp_pred, volume_fbp-volume_fbp_rs)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # print('max memory allocated:', torch.cuda.max_memory_allocated(device=self.device)//2**20, 'MiB')
                print('epoch:', epoch, '/ num:', i, '/ iter:', numIter, '/ loss:', loss.item()/25)

            ### SAVING... ###
            self.model_save(epoch)
            fbp_pred = fbp_pred[0,0,:,30,:].detach().squeeze(0).cpu().numpy()
            volume_fbp_rs = volume_fbp_rs[0,0,:,30,:].detach().squeeze(0).cpu().numpy()
            volume_fbp = volume_fbp[0,0,:,30,:].detach().squeeze(0).cpu().numpy()
            png = np.concatenate((norm(volume_fbp),
                                    norm(fbp_pred),
                                    norm(volume_fbp-volume_fbp_rs), 
                                    norm(volume_fbp-fbp_pred),
                                    norm(volume_fbp_rs)), axis=1)
            imageio.imsave(os.path.join(self.config['MODEL_SAVE_PATH'], 'epoch'+str(epoch)+'_img.png'), png)

    def train2d(self):
        def norm(x):
            return (x-np.min(x))/(np.max(x)-np.min(x))
            
        for epoch in range(self.config['NUM_EPOCH']):
            for i, data in enumerate(self.train_loader):
                numIter = epoch*len(self.train_loader)+i
                # print('epoch', epoch, 'num:', i, 'iter:', numIter)
                proj = data['proj'].to(self.device)
                proj_rs = data['proj_rs'].to(self.device)
                # print(proj.shape, proj_rs.shape)

                res_pred = self.model(proj)
                loss = 10*self.l1loss(res_pred, proj-proj_rs)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # print('max memory allocated:', torch.cuda.max_memory_allocated(device=self.device)//2**20, 'MiB')
                print('epoch:', epoch, '/ num:', i, '/ iter:', numIter, '/ loss:', loss.item()/10)

            ### SAVING... ###
            self.model_save(epoch)
            res_pred = res_pred[0,0].detach().squeeze(0).cpu().numpy()
            proj_rs = proj_rs[0,0].detach().squeeze(0).cpu().numpy()
            proj = proj[0,0].detach().squeeze(0).cpu().numpy()
            png = np.concatenate((norm(proj),
                                    norm(res_pred),
                                    norm(proj-proj_rs), 
                                    norm(proj-res_pred),
                                    norm(proj_rs)), axis=1)
            imageio.imsave(os.path.join(self.config['MODEL_SAVE_PATH'], 'epoch'+str(epoch)+'_img.png'), png)

    def train_merge(self):
        def norm(x):
            return (x-np.min(x))/(np.max(x)-np.min(x))
            
        for epoch in range(self.config['NUM_EPOCH']):
            for i, data in enumerate(self.train_loader):
                numIter = epoch*len(self.train_loader)+i
                # print('epoch', epoch, 'num:', i, 'iter:', numIter)
                volume_fbp_rs = data['volume_fbp_rs'].to(self.device)
                volume_fbp = data['volume_fbp'].to(self.device)
                volume_fbp_res_2d = data['volume_fbp_res_2d'].to(self.device)
                volume_fbp_res_3d = data['volume_fbp_res_3d'].to(self.device)
                # print(proj.shape, proj_rs.shape)
                
                volume_fbp_res = torch.cat((volume_fbp_res_2d, volume_fbp_res_3d), dim=1)
                fbp_pred = self.model(volume_fbp_res)+volume_fbp_res_3d
                loss = 50*self.l1loss(fbp_pred, volume_fbp-volume_fbp_rs)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loss_2d = self.l1loss(volume_fbp_res_2d, volume_fbp-volume_fbp_rs).item()
                loss_3d = self.l1loss(volume_fbp_res_3d, volume_fbp-volume_fbp_rs).item()
                # print('max memory allocated:', torch.cuda.max_memory_allocated(device=self.device)//2**20, 'MiB')
                print('epoch:', epoch, '/ num:', i, '/ iter:', numIter, '/ loss:', loss.item()/50, loss_2d, loss_3d)

            ### SAVING... ###
            self.model_save(epoch)
            fbp_pred = fbp_pred[0,0,:,30,:].detach().squeeze(0).cpu().numpy()
            volume_fbp_rs = volume_fbp_rs[0,0,:,30,:].detach().squeeze(0).cpu().numpy()
            volume_fbp = volume_fbp[0,0,:,30,:].detach().squeeze(0).cpu().numpy()
            png = np.concatenate((norm(volume_fbp),
                                    norm(fbp_pred),
                                    norm(volume_fbp-volume_fbp_rs), 
                                    norm(volume_fbp-fbp_pred),
                                    norm(volume_fbp_rs)), axis=1)
            imageio.imsave(os.path.join(self.config['MODEL_SAVE_PATH'], 'epoch'+str(epoch)+'_img.png'), png)

def ges_Aonfig(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

if __name__ == '__main__':
    config = ges_Aonfig('config.yaml')
    cudnn.benchmark = True
    m = Train(config)
    if config['MODE'] == 'train':
        if config['TRAIN_MERGE']:
            m.train_merge()
        else:
            if config['DIM'] == 3:
                m.train3d()
            elif config['DIM'] == 2:
                m.train2d()
    elif config['MODE'] == 'test':
        m.test()