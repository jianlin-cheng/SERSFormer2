
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from pandas import read_csv
import re
import scipy.signal as signal
DATASET_DIR = "./"
print(DATASET_DIR)
label_dict = {'No_pest_present':0,'carbophenothion':1,'coumaphos':2,'oxamyl':3,'phosmet':4,'thiabendazole':5}
Num_classes = len(label_dict)

train = glob.glob("./Spinach_Strawbery/**/**/**/**/*.*", recursive=False)


print("Number of Train pest samples => ", len(train))



class SERSDatav3(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(train)
    
    def __getitem__(self, idx):
        pest_data = train[idx].replace("\\","/")
        
        #print(pest_data)
        #pest_file_name = os.path.basename(pest_data)
        #pest_label = os.path.basename(os.path.dirname(pest_data))
        
        path_parts = os.path.dirname(pest_data).split('/')

        if path_parts[-1] == "No_pest_present":
            label_class_list = [1,0,0,0,0,0]
            label_conc_list = [0.0,0.0,0.0,0.0,0.0,0.0]
        else:
            ratio = path_parts[-3].split('_')
            pest_list = path_parts[-2].split('_')
            
            concentration = float(re.findall(r"[-+]?\d*\.\d+|\d+", path_parts[-1])[0])*(float(ratio[0])/100)
            label_class_list = [1 if i in pest_list else 0 for i in label_dict.keys()]
            label_conc_list = [1*concentration if i in pest_list else 0 for i in label_dict.keys()]
            
        pest_label1 = torch.as_tensor(label_class_list)
        #print(pest_label1)
        #pest_concen = float(re.findall(r"[-+]?\d*\.\d+|\d+", pest_data)[0])  if pest_label1!=0 else 0.0
        pest_concen = torch.as_tensor(label_conc_list)
        pest_intensity = read_csv(pest_data,header=None, index_col=0).T
        pest_intensity_np = np.array(pest_intensity).squeeze(0)
        pest_intensity_np = np.array([np.log1p(p) if p >1 else 0.001 for p in pest_intensity_np ])
        #pest_intensity_np = np.array(np.log1p(pest_intensity_np))
        #pest_intensity_tensor = torch.from_numpy(pest_intensity_np).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #norm_pest_intensity_tensor_withWave = torch.row_stack((norm_pest_intensity_tensor,torch.as_tensor(pest_intensity.columns))).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.mean()) / pest_intensity_tensor.std() 
        pest_95 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,95) else 0.0 for x in pest_intensity_np]).float()
        pest_85 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,85) else 0.0 for x in pest_intensity_np]).float()
        pest_75 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,75) else 0.0 for x in pest_intensity_np]).float()
        pest_50 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,50) else 0.0 for x in pest_intensity_np]).float()
        win = signal.windows.hann(32)
        intensity_tensor = (signal.convolve(pest_intensity_np, win, mode='same',method='direct') / sum(win))
        conv_pest_int = torch.from_numpy(intensity_tensor).unsqueeze(0).float()
        pest_intensity_tensor = torch.from_numpy(pest_intensity_np).unsqueeze(0).float()
        norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #norm_pest_intensity_tensor = torch.from_numpy(pest_intensity_np).unsqueeze(0).float()
        stacked_tensors = torch.row_stack((pest_95,pest_85,pest_75,pest_50))
        return [pest_label1,pest_concen, norm_pest_intensity_tensor,stacked_tensors,conv_pest_int]   


test = glob.glob("./TestData/**/**/**/**/*.*", recursive=False)
print("Number of Test pest samples => ", len(test))
class SERSDatav3test(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(test)
    
    def __getitem__(self, idx):
        pest_data = test[idx].replace("\\","/")
        
        #print(pest_data)
        #pest_file_name = os.path.basename(pest_data)
        #pest_label = os.path.basename(os.path.dirname(pest_data))
        
        path_parts = os.path.dirname(pest_data).split('/')

        if path_parts[-1] == "No_pest_present":
            label_class_list = [1,0,0,0,0,0]
            label_conc_list = [0.0,0.0,0.0,0.0,0.0,0.0]
        else:
            ratio = path_parts[-3].split('_')
            pest_list = path_parts[-2].split('_')
            
            concentration = float(re.findall(r"[-+]?\d*\.\d+|\d+", path_parts[-1])[0])*(float(ratio[0])/100)
            label_class_list = [1 if i in pest_list else 0 for i in label_dict.keys()]
            label_conc_list = [1*concentration if i in pest_list else 0 for i in label_dict.keys()]
            
        pest_label1 = torch.as_tensor(label_class_list)
        #print(pest_label1)
        #pest_concen = float(re.findall(r"[-+]?\d*\.\d+|\d+", pest_data)[0])  if pest_label1!=0 else 0.0
        pest_concen = torch.as_tensor(label_conc_list)
        pest_intensity = read_csv(pest_data,header=None, index_col=0).T
        pest_intensity_np = np.array(pest_intensity).squeeze(0)
        pest_intensity_np = np.array([np.log1p(p) if p >1 else 0.001 for p in pest_intensity_np ])
        #pest_intensity_np = np.array(np.log1p(pest_intensity_np))
        #pest_intensity_tensor = torch.from_numpy(pest_intensity_np).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #norm_pest_intensity_tensor_withWave = torch.row_stack((norm_pest_intensity_tensor,torch.as_tensor(pest_intensity.columns))).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.mean()) / pest_intensity_tensor.std() 
        pest_95 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,95) else 0.0 for x in pest_intensity_np]).float()
        pest_85 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,85) else 0.0 for x in pest_intensity_np]).float()
        pest_75 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,75) else 0.0 for x in pest_intensity_np]).float()
        pest_50 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,50) else 0.0 for x in pest_intensity_np]).float()
        win = signal.windows.hann(32)
        intensity_tensor = (signal.convolve(pest_intensity_np, win, mode='same',method='direct') / sum(win))
        conv_pest_int = torch.from_numpy(intensity_tensor).unsqueeze(0).float()
        pest_intensity_tensor = torch.from_numpy(pest_intensity_np).unsqueeze(0).float()
        norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #norm_pest_intensity_tensor = torch.from_numpy(pest_intensity_np).unsqueeze(0).float()
        stacked_tensors = torch.row_stack((pest_95,pest_85,pest_75,pest_50))
        return [pest_label1,pest_concen, norm_pest_intensity_tensor,stacked_tensors,conv_pest_int]   


#da = SERSDatav3(DATASET_DIR)
#print(da[37])
