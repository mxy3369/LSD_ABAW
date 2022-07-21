import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

from torch.utils.data import DataLoader
from util.DatasetDefinition import MTL_DataSet
from util.ModelDefinition import BasicNet
from util.HelperFunction import model_fit, model_predict1, model_predict_joint, model_predict_hand, test, test_joint
from util.utils import get_all_data_loaders, get_config, get_MultiPIE_dataloaders, get_S_dataloaders, resplit_s_data

import warnings
warnings.filterwarnings('ignore')

#添加配置
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

config = get_config(opts.config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader, test_hand_loader = get_S_dataloaders(config, 'test')
model_hand = BasicNet(output_size=2).to(device)
model = BasicNet(output_size=6).to(device)
model_hand.load_state_dict(torch.load(os.path.join(config['model_root'], "hand_best_model.pkl")))
model.load_state_dict(torch.load(os.path.join(config['model_root'], "mouse_656.pkl")))
save_path = os.path.join(config['data_root'], 'result', 'mouse.txt')
# test(model, test_loader, device, save_path)
test_joint(model, model_hand, test_loader, test_hand_loader, device, save_path)