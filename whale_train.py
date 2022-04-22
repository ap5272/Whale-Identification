import os
import warnings
from pprint import pprint
from glob import glob
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import torchvision.transforms as T
from box import Box
import math

import timm
from timm import create_model
from sklearn.model_selection import StratifiedKFold
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule



config = {'seed': 2021,
          'root': '', 
          'n_splits': 2,
          'epoch': 1,
          'trainer': {
              'gpus': 1,
              'accumulate_grad_batches': 1,
              'auto_scale_batch_size': "binsearch",
              'auto_lr_find': True,
              'progress_bar_refresh_rate': 1,
              'fast_dev_run': False,
              'num_sanity_val_steps': 0,
              'resume_from_checkpoint': None,
          },
          'transform':{
              'name': 'get_default_transforms',
              'image_size': 224
          },
          'train_loader':{
              'batch_size': 64,
              'shuffle': True,
              'num_workers': 4,
              'pin_memory': False,
              'drop_last': True,
          },
          'val_loader': {
              'batch_size': 64,
              'shuffle': False,
              'num_workers': 4,
              'pin_memory': False,
              'drop_last': False
         },
          'model':{
              'name': 'swin_small_patch4_window7_224',
              'output_dim': 15587,
              'batch_size': 64
          },
          'optimizer':{
              'name': 'optim.AdamW',
              'params':{
                  'lr': 1e-4
              },
          },
          'scheduler':{
              'name': 'optim.lr_scheduler.CosineAnnealingWarmRestarts',
              'params':{
                  'T_0': 20,
                  'eta_min': 1e-2,
              }
          },
          'loss': 'nn.BCEWithLogitsLoss',
         'ArcFace':{
             'embedding_size': 512,
             's': 30,
             'm': 0.5,
         },

}



class WhaleDataset(Dataset):
    def __init__(self, df, image_size=224):
        self._X = df["image"].values
        self._y = df["indiv"].values
        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
        image = self._transform(image)
        
        label = torch.zeros(15587)
        label[self._y[idx]] = 1.0
        
        return image, label



class WhaleDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            WhaleDataset(self._train_df, self._cfg.transform.image_size)
            if train
            else WhaleDataset(self._val_df, self._cfg.transform.image_size)
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)




def get_default_transforms():
    transform = {
        "train": T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ConvertImageDtype(torch.float),
                #T.RandomPosterize(bits=2),
                #T.RandomPerspective(distortion_scale=0.6, p=1.0),
                #T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        "val": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                #T.Resize
                #T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    }
    return transform




class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features=2048,
        out_features=15587,
        s=30,
        m=0.5,
        easy_margin=False,
        ls_eps=0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        

    def forward(self, inpt, label, device):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(inpt).to(device), F.normalize(self.weight).to(device))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        
        one_hot = torch.zeros(cosine.size()).to(device=device)
        
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        #one_hot.scatter_(1, label.long().to(device), 1)
        
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output




class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.arc_s = cfg.ArcFace.s
        self.arc_m = cfg.ArcFace.m #threshold
        self.embedding_size = cfg.ArcFace.embedding_size
        self.arc_ls_eps=0
        self.arc_easy_margin = False
        
        self.__build_model()
        
        self._criterion = eval(self.cfg.loss)()
        
        self.transform = get_default_transforms()
        self.save_hyperparameters(cfg)
        

    def __build_model(self):
        self.backbone = create_model(
            self.cfg.model.name, 
            pretrained=True, 
            num_classes=0, 
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        # new head to utilize pretrained model
        
        self.embedding = nn.Linear(num_features, self.embedding_size)
        
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

        self.arc = ArcMarginProduct(
            in_features=self.embedding_size,
            out_features=self.cfg.model.output_dim,
            s=self.arc_s,
            m=self.arc_m,
            easy_margin=self.arc_easy_margin,
            ls_eps=self.arc_ls_eps,
        )
        

    def forward(self, x):
        f = self.backbone(x)
        out = self.embedding(f)
        return out
    

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, 'train')
        return {'loss': loss, 'pred': pred, 'labels': labels}
        
    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, 'val')
        return {'pred': pred, 'labels': labels}
    
    
    def __share_step(self, batch, mode):
        ### add ARC in step ###
        images, labels = batch
        labels = labels.float()
        images = self.transform[mode](images)
        logits = self.forward(images).squeeze(1)
        
        # labels are just indiv number, not vector and should be between 0 and 1
        dd ="cuda" if torch.cuda.is_available() else "cpu"
        
        out = self.arc(logits, labels.argmax(dim=1)/15587, dd)
        
        loss = self._criterion(out, labels/15587)
        
        pred = logits.sigmoid().detach().cpu()
        labels = labels.detach().cpu()
        print(f"{mode}:{loss.item()}")
        return loss, pred, labels
        
        
    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'val')    
        
    def __share_epoch_end(self, outputs, mode):
        ### do inference here ###
        
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['pred'], out['labels']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        
        ### use topk 5 as metric
        top5 = torch.topk(preds,5).indices
        
        metrics = (top5 == labels.argmax(dim=1).unsqueeze(1)).any(1).float().mean()
        
        print(f'{mode}_loss', metrics)
        
        self.log(f'{mode}_loss', metrics)
    


    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer,
            **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]




if __name__ == "__main__":
    
    config = Box(config)
    
    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)


    df = pd.read_csv(os.path.join("train.csv"))
    df = df.assign(indiv=(df.individual_id ).astype('category').cat.codes)

    df.species.replace({"globis": "short_finned_pilot_whale",
                              "pilot_whale": "short_finned_pilot_whale",
                              "kiler_whale": "killer_whale",
                              "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)

    df["image"] = df["image"].apply(lambda x: os.path.join(config.root, "train_images", x))


    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )


    for fold, (train_idx, val_idx) in enumerate(skf.split(df["image"], df["indiv"])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = WhaleDataModule(train_df, val_df, config)
        model = Model(config)
        
        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        logger = TensorBoardLogger(config.model.name)
        
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config.trainer,
        )
        
        #trainer.tune(model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
        
        
