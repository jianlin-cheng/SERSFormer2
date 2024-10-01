"""
author: Akshata 
timestamp: Tue Oct 01 2024 12:10 PM
"""

import os
import glob
from typing import List, Union
from lightning.pytorch.utilities.types import EPOCH_OUTPUT
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv, DataFrame
from torch import  Tensor
import torch.nn.functional as F
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score,MultilabelF1Score,MultilabelAccuracy,MultilabelPrecision,MultilabelRecall,MultilabelConfusionMatrix
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError,SpearmanCorrCoef
from torchmetrics import MeanMetric
from scipy.stats import gaussian_kde
from Model_v2 import TransformerClassifyRegress_sep
from argparse import ArgumentParser
import scipy.signal as signal
from Dataset import SERSDatav3,SERSDatav3test
from Metric import MultiLabelUnifiedMetric,CustomMetricCollection
AVAIL_GPUS = [0]
NUM_NODES = 1
BATCH_SIZE = 64
DATALOADERS = 4
ACCELERATOR = 'gpu'
EPOCHS = 20
ATT_HEAD = 4
ENCODE_LAYERS = 4
DATASET_DIR = "./"

label_dict = {'No_pest_present':0,'carbophenothion':1,'coumaphos':2,'oxamyl':3,'phosmet':4,'thiabenzadole':5}
Num_classes = len(label_dict)
torch.set_default_tensor_type(torch.FloatTensor)  # Ensure that the default tensor type is FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""

torch.set_default_tensor_type(torch.FloatTensor)  # Ensure that the default tensor type is FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose the device you want to use

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner to find the best algorithm to use for hardware
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Set the default tensor type to CUDA FloatTensor
    torch.set_float32_matmul_precision('medium')  # Set Tensor Core precision to medium

"""

CHECKPOINT_PATH = f"{DATASET_DIR}/Training2/SERSFOrmer2_0_multireg_multiclass_20epoch_DWA"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


class SERSClassifyRegress(pl.LightningModule):
    def __init__(self, learning_rate=1e-4,attn_head=ATT_HEAD,encoder_layers=ENCODE_LAYERS,n_labels=1, **model_kwargs):
        super().__init__()
        
        self.num_outputs = 1
        self.save_hyperparameters()
        
        self.model = TransformerClassifyRegress_sep(attn_head=attn_head,encoder_layers=encoder_layers,n_labels=n_labels,**model_kwargs)
        self.loss_fn = nn.BCELoss()
        self.loss_fn_reg = nn.MSELoss()
        self.metrics_class = MetricCollection([MultilabelAccuracy(num_labels=n_labels),
                                         MultilabelPrecision(num_labels=n_labels),
                                         MultilabelRecall(num_labels=n_labels),
                                         MultilabelF1Score(num_labels=n_labels)])
        self.metrics_regress = MetricCollection({ 
                                                'MSE':MeanSquaredError(num_outputs=n_labels),
                                                 'R2Score':R2Score(num_outputs=n_labels)
                                                 })
        self.metric = MultiLabelUnifiedMetric(num_labels=n_labels).to(device)
        self.metric = self.metric.cuda()
        self.metrics_regress2 = MetricCollection([self.metric]).cuda()
        self.train_metrics_class = self.metrics_class.clone(prefix="train_")
        self.train_metrics_regress = self.metrics_regress.clone(prefix="train_")
        self.train_metrics_regress2 = self.metrics_regress2.clone(prefix="train_")
        self.valid_metrics_class = self.metrics_class.clone(prefix="valid_")
        self.valid_metrics_regress = self.metrics_regress.clone(prefix="valid_")
        self.valid_metrics_regress2 = self.metrics_regress2.clone(prefix="valid_")
        self.test_metrics_class = self.metrics_class.clone(prefix="test_")
        self.test_metrics_regress = self.metrics_regress.clone(prefix="test_")
        self.test_metrics_regress2 = self.metrics_regress2.clone(prefix="test_")
        self.step_scheduler_after = "epoch"
        self.metrics = "valid_loss"
        self.test_step_class_pred = []
        self.test_step_class_tar = []
        self.test_step_reg_pred = []
        self.test_step_reg_tar = []
        self.T = 2  # Temperature for softmax
        self.loss_histories = [[] for _ in range(2)] 
    def forward(self, pest_sample):
        
        x = self.model(pest_sample)
        return x
    def compute_dwa_weights(self):
        if len(self.loss_histories[0]) > 1:
            r1 = self.loss_histories[0][-1] / self.loss_histories[0][-2]
            r2 = self.loss_histories[1][-1] / self.loss_histories[1][-2]

            weights = np.exp([r1 / self.T, r2 / self.T])
            weights /= weights.sum()

            weight_task1 = weights[0]
            weight_task2 = weights[1]
        else:
            weight_task1 = 0.5
            weight_task2 = 0.5

        return weight_task1, weight_task2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=20, eps=1e-10,verbose=True)
        metric_to_track = 'valid_loss'
        return{'optimizer':optimizer,
               'lr_scheduler':lr_scheduler,
                'monitor':metric_to_track,
               }
    def lr_scheduler_step(self, scheduler, metric,*args, **kwargs):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
    
    def training_step(self,batch,batch_idx):
        batch_data = batch[2:]
        
        y_hat = self.forward(batch_data)
        batch_label_class = batch[0].type('torch.LongTensor')
        
        batch_label_class = batch_label_class[:,None].cuda()
        batch_conc = batch[1].to(y_hat[1].dtype).cuda()
        #batch_conc = batch_conc[:,None]
        class_pred = y_hat[0]
        
        conc_pred = y_hat[1]
       
        loss_class = self.loss_fn(class_pred,batch_label_class.float().squeeze())
        metric_log_class = self.train_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss_reg = self.loss_fn_reg(conc_pred,batch_conc)
        metric_log_reg = self.train_metrics_regress(conc_pred, batch_conc.float())
        metric_log_reg2 = self.train_metrics_regress2(conc_pred, batch_conc.float(),batch_label_class.int().squeeze())
        self.log_dict({list(metric_log_reg.keys())[0]:metric_log_reg[list(metric_log_reg.keys())[0]].mean(),list(metric_log_reg.keys())[1]:metric_log_reg[list(metric_log_reg.keys())[1]].mean()})
        self.log_dict(metric_log_reg2)
        #self.log('train_loss_class', loss_class, on_step=True, on_epoch=True, sync_dist=True)
        #self.log('train_loss_reg', loss_reg, on_step=True, on_epoch=True, sync_dist=True)

        self.loss_histories[0].append(loss_class.item())
        self.loss_histories[1].append(loss_reg.item())

        weight_task1, weight_task2 = self.compute_dwa_weights()

        loss = weight_task1 * loss_class + weight_task2 * loss_reg
        self.log('train_loss_class', loss_class, prog_bar=True,on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_reg', loss_reg, prog_bar=True,on_step=True, on_epoch=True, sync_dist=True)
        self.log('total_loss', loss, prog_bar=True,on_step=True, on_epoch=True, sync_dist=True)
        self.log('weight_task1', weight_task1, prog_bar=True,on_step=True, on_epoch=True, sync_dist=True)
        self.log('weight_task2', weight_task2, prog_bar=True,on_step=True, on_epoch=True, sync_dist=True)
        
        #self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self,batch,batch_idx):
        batch_data = batch[2:]
        
        y_hat = self.forward(batch_data)
        batch_label_class = batch[0].type('torch.LongTensor')
        batch_label_class = batch_label_class[:,None].cuda()
        batch_conc = batch[1].to(y_hat[1].dtype).cuda()
        #print(batch_conc.shape)
        #batch_conc = batch_conc[:,None]
        class_pred = y_hat[0]
        
        conc_pred = y_hat[1]
        #print(conc_pred.shape)
        loss_class = self.loss_fn(class_pred,batch_label_class.float().squeeze())
        metric_log_class = self.valid_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss_reg = self.loss_fn_reg(conc_pred,batch_conc)
        metric_log_reg = self.valid_metrics_regress(conc_pred, batch_conc.float())
        metric_log_reg2 = self.valid_metrics_regress2(conc_pred, batch_conc.float(),batch_label_class.int().squeeze())
        self.log_dict({list(metric_log_reg.keys())[0]:metric_log_reg[list(metric_log_reg.keys())[0]].mean(),list(metric_log_reg.keys())[1]:metric_log_reg[list(metric_log_reg.keys())[1]].mean()})
        self.log_dict(metric_log_reg2)
        self.log('valid_loss_class', loss_class, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss_reg', loss_reg, on_step=True, on_epoch=True, sync_dist=True)
        loss = (loss_class+loss_reg)
        self.log('valid_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        
       
    
    def test_step(self,batch, batch_idx):
        batch_data = batch[2:]
        y_hat = self.forward(batch_data)
        batch_label_class = batch[0].type('torch.LongTensor').cuda()
        batch_label_class = batch_label_class[:,None]
        batch_conc = batch[1].to(y_hat[1].dtype).cuda()
        #batch_conc = batch_conc[:,None]
        class_pred = y_hat[0]
        conc_pred = y_hat[1]
        loss_class = self.loss_fn(class_pred,batch_label_class.float().squeeze())
        metric_log_class = self.test_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss_reg = self.loss_fn_reg(conc_pred,batch_conc)
        metric_log_reg = self.test_metrics_regress(conc_pred, batch_conc.float())
        metric_log_reg2 = self.test_metrics_regress2(conc_pred, batch_conc.float(),batch_label_class.int().squeeze())
        self.log_dict({list(metric_log_reg.keys())[0]:metric_log_reg[list(metric_log_reg.keys())[0]].mean(),list(metric_log_reg.keys())[1]:metric_log_reg[list(metric_log_reg.keys())[1]].mean()})
        self.log_dict(metric_log_reg2)
        print("Test Data Confusion Matrix: \n") 
        self.log('test_loss_class', loss_class, on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_loss_reg', loss_reg, on_step=True, on_epoch=True, sync_dist=True)
        loss = (loss_class+loss_reg)/2
        
        self.log('test_loss',loss, on_epoch=True, sync_dist=True)
        self.test_step_class_tar.append(batch_label_class.squeeze())
        self.test_step_class_pred.append(class_pred)
        self.test_step_reg_pred.append(conc_pred)
        self.test_step_reg_tar.append(batch_conc)
        
        return {f'preds_class' : class_pred, f'targets_class' : batch_label_class.squeeze(),f'preds_reg':conc_pred,f'targets_reg':batch_conc}
        
          
    
    def test_epoch_end(self, outputs,*arg, **kwargs):
        # Log individual results for each dataset
        
        #for i  in range(len(outputs)):
            #dataset_outputs = outputs
            #torch.save(dataset_outputs,"Predictions.pt")
            #class_preds = torch.cat([x[f'preds_class'] for x in dataset_outputs])
            #class_targets = torch.cat([x[f'targets_class'] for x in dataset_outputs])
            class_preds = self.test_step_class_pred
            print(class_preds)
            class_targets = self.test_step_class_tar
            conf_mat = MultilabelConfusionMatrix(num_labels=Num_classes,)
            conf_vals = conf_mat(class_preds, class_targets)
            fig, ax = plt.subplots()
            sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            wandb.log({f"Confusion Matrix" :wandb.Image(fig)})

            #reg_preds = torch.cat([x[f'preds_reg'] for x in dataset_outputs])
            #reg_targets = torch.cat([x[f'targets_reg'] for x in dataset_outputs])
            reg_preds = self.test_step_reg_pred
            reg_targets = self.test_step_reg_tar
            data = [[x, y] for (x, y) in zip(reg_targets, reg_preds)]
            reg_preds_np = reg_preds.squeeze().numpy()
            reg_targets_np = reg_targets.squeeze().numpy()

            df = DataFrame({'True': reg_targets_np, 'Pred': reg_preds_np})

            # Scatter plot
            # Calculate the point density
            xy = torch.vstack([reg_targets.T,reg_preds.T]).cpu().detach().numpy()
            
            z = gaussian_kde(xy.squeeze())(xy.squeeze())
            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = reg_targets[idx], reg_preds[idx], z[idx]
            fig1, ax = plt.subplots(figsize=(12, 6))
            plt.scatter(x,y,c=z,s=100)
            wandb.log({f"Concentration Predictions" :wandb.Image(fig1)})
            # Get unique target values
            unique_targets = df['True'].unique()
            unique_targets = sorted(unique_targets)
            print(unique_targets)
            # Set up subplots
            num_targets = len(unique_targets)
            fig2, axes = plt.subplots(nrows=1, ncols=num_targets, figsize=(15, 5),sharey=True)

            # Create violin plots for each target value
            for i, target_value in enumerate(unique_targets):
                target_df = df[df['True'] == target_value]
                sns.violinplot(x='True', y='Pred', data=target_df, inner="sticks", color="lightgreen",cut=0 ,ax=axes[i])

                axes[i].set_title(f'Target = {target_value}')

            plt.tight_layout()
            plt.xlabel("True Concentration")
            plt.ylabel("Predicted Concentration")
            plt.title("True and Predicted Concentration")
            plt.tight_layout()
            wandb.log({f"Kernel Density Estimation of Concentrations" :wandb.Image(fig2)})
            return super().test_epoch_end(outputs)
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--attn_head',type=int,default=ATT_HEAD)
        parser.add_argument('--encoder_layers',type=int,default=ENCODE_LAYERS)
        parser.add_argument('--n_class',type=int,default=1)
        parser.add_argument('--entity_name', type=str,default=None, help="Weights and Biases entity name")
        return parser


def train_pesticide_classifier():
    pl.seed_everything(42, workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SERSClassifyRegress.add_model_specific_args(parser)
    parser.add_argument('--num_gpus', type=int, default=AVAIL_GPUS,
                        help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--nodes', type=int, default=NUM_NODES, help="Number of nodes to use")
    parser.add_argument('--num_epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                        help="effective_batch_size = batch_size * num_gpus * num_nodes")
    parser.add_argument('--num_dataloader_workers', type=int, default=DATALOADERS)
    
    parser.add_argument('--project_name', type=str, default='SERSFormer2_0',
                        help="Weights and Biases project name")
    parser.add_argument('--save_dir', type=str, default=CHECKPOINT_PATH, help="Directory in which to save models")

    parser.add_argument('--unit_test', type=int, default=False,
                        help="helps in debug, this touches all the parts of code."
                             "Enter True or num of batch you want to send, " "eg. 1 or 7")
    args = parser.parse_args()
    
    args.devices = args.num_gpus
    args.num_nodes = args.nodes
    args.accelerator = ACCELERATOR
    args.max_epochs = args.num_epochs
    args.fast_dev_run = args.unit_test
    args.log_every_n_steps = 1
    args.detect_anomaly = True
    args.enable_model_summary = True
    args.weights_summary = "full"
    
    save_PATH = DATASET_DIR+"/Training2/"+args.save_dir
    os.makedirs(save_PATH, exist_ok=True)

    dataset = SERSDatav3(DATASET_DIR)
    dataset_test = SERSDatav3test(DATASET_DIR)
    print(len(dataset))
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - (train_size)
    #test_size = len(dataset) - (train_size+val_size)
    dataset_train,dataset_valid = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # using validation data for testing here
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_dataloader_workers)
    print(train_size)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    #test_loader2 = DataLoader(dataset=dataset_test2,batch_size=BATCH_SIZE,shuffle=False,num_workers=args.num_dataloader_workers)
    torch.save(test_loader,DATASET_DIR+'/'+args.save_dir+'_testComplete.pt')
    model = SERSClassifyRegress(learning_rate=1e-4,n_labels=Num_classes,attn_head=args.attn_head,encoder_layers=args.encoder_layers)
    
    trainer = pl.Trainer().from_argparse_args(args)
    
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=10, dirpath=save_PATH, filename='pesticides_classify_{epoch:02d}_{valid_loss:6f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=10)
    trainer.callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir, offline=False, save_dir=".",sync_tensorboard=True)
    
    trainer.logger = logger
    
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    
   



if __name__ == "__main__":
    
    train_pesticide_classifier()
    #wandb.finish()
