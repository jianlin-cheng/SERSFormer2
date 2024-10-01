"""
author: Akshata 
timestamp: July 28 2024 08:10 PM
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
AVAIL_GPUS = [2]
NUM_NODES = 1
BATCH_SIZE = 64
DATALOADERS = 16
ACCELERATOR = 'gpu'
EPOCHS = 4
ATT_HEAD = 4
ENCODE_LAYERS = 4
DATASET_DIR = "./"
from Metric import MultiLabelUnifiedMetric
label_dict = {'No_pest_present':0,'carbophenothion':1,'coumaphos':2,'oxamyl':3,'phosmet':4,'thiabenzadole':5}
Num_classes = len(label_dict)
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
        self.model
        self.loss_fn = nn.BCELoss()
        self.loss_fn_reg = nn.MSELoss()
        self.metrics_class = MetricCollection([MultilabelAccuracy(num_labels=n_labels),
                                         MultilabelPrecision(num_labels=n_labels),
                                         MultilabelRecall(num_labels=n_labels),
                                         MultilabelF1Score(num_labels=n_labels)])
        self.metrics_regress = MetricCollection([ MeanSquaredError(num_outputs=n_labels),
                                                 R2Score(num_outputs=n_labels)])
        self.metric = MultiLabelUnifiedMetric(num_labels=n_labels)
        self.metric = self.metric
        self.metrics_regress2 = MetricCollection([self.metric])
        self.test_metrics_class = self.metrics_class.clone(prefix="test_")
        self.test_metrics_regress = self.metrics_regress.clone(prefix="test_")
        self.test_metrics_regress2 = self.metrics_regress2.clone(prefix="test_")
        self.step_scheduler_after = "epoch"
        self.metrics = "valid_loss"
        self.test_step_class_pred = []
        self.test_step_class_tar = []
        self.test_step_reg_pred = []
        self.test_step_reg_tar = []

    def forward(self, pest_sample):
        
        x = self.model(pest_sample)
        return x

    
    def test_step(self,batch, batch_idx):
        batch_data = batch[2:]
        y_hat = self.forward(batch_data)
        batch_label_class = batch[0].type('torch.LongTensor')
        batch_label_class = batch_label_class[:,None]
        batch_conc = batch[1].to(y_hat[1].dtype)
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
        loss = (loss_class+loss_reg)
        
        self.log('test_loss',loss, on_epoch=True, sync_dist=True)
        self.test_step_class_tar.append(batch_label_class.squeeze())
        self.test_step_class_pred.append(class_pred)
        self.test_step_reg_pred.append(conc_pred)
        self.test_step_reg_tar.append(batch_conc)
        
        return {f'preds_class' : class_pred, f'targets_class' : batch_label_class.squeeze(),f'preds_reg':conc_pred,f'targets_reg':batch_conc}
        
          
    
    def test_epoch_end(self,outputs):
        # Log individual results for each dataset
        
        #for i  in range(len(outputs)):
            dataset_outputs = outputs
            torch.save(dataset_outputs,"Predictions_ofnormaltest.pt")
            class_preds = torch.cat([x[f'preds_class'] for x in dataset_outputs])
            class_targets = torch.cat([x[f'targets_class'] for x in dataset_outputs])
            #class_preds = self.test_step_class_pred
            print(class_preds)
            #class_targets = self.test_step_class_tar
            conf_mat = MultilabelConfusionMatrix(num_labels=Num_classes)
            conf_vals = conf_mat(class_preds, class_targets)
            fig, ax = plt.subplots()
            # Plot confusion matrices using Seaborn
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, cm in enumerate(conf_vals.cpu()):
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
                axes[i].set_title(f'Label {i+1}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('True')

            #sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            wandb.log({f"Confusion Matrix" :wandb.Image(fig)})

            #reg_preds = torch.cat([x[f'preds_reg'] for x in dataset_outputs])
            #reg_targets = torch.cat([x[f'targets_reg'] for x in dataset_outputs])
           
            return super().test_epoch_end(outputs)
    def predict_step(self,batch, batch_idx):
        batch_data = batch[2:]
        return self(batch_data)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--attn_head',type=int,default=ATT_HEAD)
        parser.add_argument('--encoder_layers',type=int,default=ENCODE_LAYERS)
        parser.add_argument('--n_class',type=int,default=1)
        parser.add_argument('--entity_name', type=str,default=None, help="Weights and Biases entity name")
        parser.add_argument('--save_dir', type=str, default=CHECKPOINT_PATH, help="Directory in which to save models")
        parser.add_argument('--chkpt',type=str,help="Checkpoint name")
        return parser


def train_pesticide_classifier():
    pl.seed_everything(123)
    parser = ArgumentParser()
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SERSClassifyRegress.add_model_specific_args(parser)
    
    parser.add_argument('--project_name', type=str, default='SERSClassifyRegress',
                        help="Weights and Biases project name")
    args = parser.parse_args()
    check_pt_dir = args.save_dir
    #args.accelerator = ACCELERATOR
    #dataset_test = SERSDatav3test(DATASET_DIR)
    test_loader= torch.load(DATASET_DIR+'/Training2/SERSFOrmer2_0_multireg_multiclass_test.pt')
    #test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, num_workers=DATALOADERS, shuffle=False)
    model = SERSClassifyRegress(learning_rate=1e-4,n_labels=Num_classes)
    trainer = pl.Trainer.from_argparse_args(args)
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir+"_test", offline=False, save_dir=".")
    trainer.logger = logger
    pest_checkpoint = DATASET_DIR+"/"+args.save_dir+"/"+args.chkpt
    
    trainer.test(model, dataloaders=test_loader, ckpt_path=pest_checkpoint)
    
    
   



if __name__ == "__main__":
    
    train_pesticide_classifier()
    #wandb.finish()
