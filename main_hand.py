import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

from torch.utils.data import DataLoader
from util.DatasetDefinition import MTL_DataSet
from util.ModelDefinition import BasicNet
from util.HelperFunction import model_fit, model_predict1, model_predict_joint, model_predict_hand
from util.utils import get_all_data_loaders, get_config, get_MultiPIE_dataloaders, get_S_dataloaders, resplit_s_data

import warnings
warnings.filterwarnings('ignore')

#添加配置
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the config file.')
opts = parser.parse_args()

config = get_config(opts.config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# train_loader, valid_loader, valid_hand_loader = get_S_dataloaders(config, 'HAND')
train_loader, valid_loader, _ = get_S_dataloaders(config, 'EMO')
# train_loader, valid_loader = get_MultiPIE_dataloaders(config)
test_loader, test_hand_loader = get_S_dataloaders(config, 'test')
# test_hand_loader = get_S_dataloaders(config, 'test_hand')
data = {
    'train': train_loader,
    'valid': valid_loader
    # 'test' : test_loader
}

train_option = {
    'criterion': nn.CrossEntropyLoss().to(device),
    'criterion_cpu': nn.CrossEntropyLoss(),
    'opt_class': optim.Adam,
    'weight_decay': 1e-4,
}

# model_hand = BasicNet(output_size=2).to(device)
model = BasicNet(output_size=6).to(device)
# for name, param in model.named_parameters():
#     if('layer4' in name):
#         break
#     param.requires_grad = False
model.load_state_dict(torch.load(os.path.join(config['model_root'], "pre_finetune_best_model.pkl")))
# model.load_state_dict(torch.load("/amax/mxy/com/resnet_out/model/s_630.pkl",  map_location=lambda storage, loc: storage))
# model_hand.load_state_dict(torch.load("/amax/mxy/com/resnet_out/model/hand_best_model.pkl"))
# model.load_state_dict(torch.load("/amax/mxy/com/resnet_out/model/s_630.pkl"))
save_path = os.path.join(config["model_root"], 'best_model.pkl')
model_fit(model, 5e-5, 20, data, train_option, device, save_path, print_interval=500)
