import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import accuracy_score, f1_score
# from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

import torch
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
from PIL import Image
from bisect import bisect_right
from torchmetrics import F1Score

import time
import sys
import pandas as pd



class CosineAnnealingLR_with_Restart(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        model (pytorch model): The model to save.
        out_dir (str): Directory to save snapshots
        take_snapshot (bool): Whether to save snapshots at every restart

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, model, out_dir, take_snapshot, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch

        self.model = model
        self.out_dir = out_dir
        self.take_snapshot = take_snapshot

        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2
                   for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        ## restart
        if self.current_epoch == self.Te:
            print("restart at epoch {:03d}".format(self.last_epoch + 1))

            if self.take_snapshot:
                torch.save({
                    'epoch': self.T_max,
                    'state_dict': self.model.state_dict()
                }, self.out_dir + "Weight/" + 'snapshot_e_{:03d}.pth.tar'.format(self.T_max))

            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te

def metric_acc(output, y_true):
    # output = torch.Tensor(output.detach().numpy())
    # y_true = torch.Tensor(y_true.detach().numpy())
    y_pred = output.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    # f1_score = F1Score(num_classes=6)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def plot_lr_find(lr_list, loss_list):
    lr_list, loss_list = lr_list[10:], loss_list[10:]

    begin_index = int(len(loss_list) / 3)
    min_index = np.argmin(loss_list[begin_index:])
    max_val = np.max(loss_list[:min_index])

    plot_loss_list = [x for x in loss_list if x <= max_val]
    plot_lr_list = [lr_list[i] for i, x in enumerate(loss_list) if x <= max_val]

    fig, ax = plt.subplots()
    ax.plot(plot_lr_list[:-1], plot_loss_list[:-1])
    ax.set_xscale('log')

def lr_find(model, data, train_option, device, lr_init=1e-6, beta=0.98):

    opt = train_option['opt_class'](
        filter( lambda p: p.requires_grad, model.parameters() ),
        lr=lr_init, weight_decay=train_option['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 1.1 ** epoch)

    iter_num, avg_loss, best_loss = 0, 0, float('inf')
    lr_list, loss_list = [], []
    
    os.makedirs('tmp', exist_ok=True)
    torch.save(model.state_dict(), 'tmp/lr_find_before.pkl')
    model.train()

    while True:
        iter_num += 1
        scheduler.step()
        cur_lr = opt.param_groups[0]['lr']
        lr_list.append(cur_lr)

        for X_batch, y_batch in data['train']:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            opt.zero_grad()
            out = model(X_batch)
            loss = train_option['criterion'](out, y_batch)
            loss.backward()
            opt.step()

            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            corr_avg_loss = avg_loss / (1 - beta ** iter_num) # 除以一个值，得到修正版的avg_loss
            loss_list.append(corr_avg_loss)
            best_loss = min(corr_avg_loss, best_loss)
            break


        if (iter_num - 1) % 25 == 0:
            print( 'iter_num=%d, cur_lr=%g, loss=%g' % (iter_num - 1, cur_lr, loss_list[-1]) )
        
            
        if iter_num > 10 and corr_avg_loss > 4 * best_loss:
            break

    model.load_state_dict( torch.load('tmp/lr_find_before.pkl') )
    return lr_list, loss_list

def model_predict(model, data_loader, device,epoch,save_dir):
    model.eval()
    y_true, output = [], []
    os.makedirs(save_dir +('/%d' %epoch), exist_ok=True)
    start_index = 0
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device)
            out = model(X_batch)
            output.append( out.to('cpu') )
            y_true.append(y_batch)
            X_batch = X_batch.to('cpu')
            mean = torch.Tensor([0.485,0.456,0.406]).view(1,3,1,1)
            std = torch.Tensor([0.229,0.224,0.225]).view(1,3,1,1)
            X_batch = X_batch * std + mean
            X_batch = X_batch.permute(0,2,3,1)
            X_batch = X_batch.numpy()
            X_batch = np.maximum(0, X_batch)
            X_batch = np.minimum(X_batch, 255)
            X_batch = (X_batch*255).astype(np.uint8) 
            for j in range(X_batch.shape[0]):
                X = Image.fromarray(X_batch[j]) 
                X.save(os.path.join(save_dir,('%d' %epoch),('%d.png' %start_index)))
                start_index+=1
    y_true, output = torch.cat(y_true), torch.cat(output)
    return y_true, output

def model_predict1(model, data_loader, device):
    lab2cls = {0:'ANGRER', 1:'DISGUST', 2:'FEAR', 3:'HAPPINESS', 4:'SADNESS', 5:'SURPRISE'}
    model.eval()
    y_true, output, paths = [], [], []
    start_index = 0
    with torch.no_grad():
        for i, (X_batch, y_batch, path_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device)
            # print(X_batch.size())
            out = model(X_batch)
            output.append(out.to('cpu'))
            # y_true.append(torch.LongTensor([int(((i+2))/4) for i in y_batch]))
            y_true.append(y_batch)
            paths += path_batch
    y_true, output = torch.cat(y_true), torch.cat(output)
    a = np.zeros((6,6), dtype="int")
    s = 0
    fear_output = []
    y_pred = output.argmax(axis=1)
    # tsne(output, y_true, './result/images/same_true.png')
    # tsne(output, y_pred, './result/images/same_pred.png')
    # print(y_true.size)
    for i in range(len(y_true)):
        # if y_true[i] == 2 and y_pred[i] == 2:
            # fear_output += [[output[i][j].item() for j in range(6)]]
            # with open('Fear_fail.txt', 'a') as f:
            #     f.write(paths[i] +"----------"+str(lab2cls[y_pred[i].item()])+'\n')
            # print(paths[i])
        a[y_true[i].item()][y_pred[i].item()] += 1
    #     s += 1
    # print(len(fear_output))
    # sys.exit()
    # data = pd.DataFrame(fear_output)
    # writer = pd.ExcelWriter('fear.xlsx')		# 写入Excel文件
    # data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    # writer.save()
    # writer.close()
    print(a)
    # print(s)
    acc, f1 = metric_acc(output, y_true)
    print(acc.item())
    print(f1.item())
    return y_true, output

def model_predict_joint(model, model_hand, data_loader, hand_loader, device):
    model.eval()
    model_hand.eval()
    output_hand, output_emo = [], []
    y_true, y_pred, y_pred_2 = [], [], []
    image_names = []
    lab2cls = {0:'ANGRER', 1:'DISGUST', 2:'FEAR', 3:'HAPPINESS', 4:'SADNESS', 5:'SURPRISE'}
    with torch.no_grad():
        for i, (x_data, h_data) in enumerate(zip(data_loader,hand_loader)):
            x_batch, y_batch, image_name = x_data
            h_batch, _, _ = h_data
            h_batch = h_batch.to(device)
            x_batch = x_batch.to(device)
            out_emo = model(x_batch)
            out_hand = model_hand(h_batch)
            output_hand.append(out_hand.to('cpu'))
            output_emo.append(out_emo.to('cpu'))
            y_true.append(y_batch)
            image_names += image_name
    y_true, output_hand, output_emo = torch.cat(y_true), torch.cat(output_hand), torch.cat(output_emo)
    hand_pre = output_hand.argmax(axis=1)
    emo_pre = output_emo.argmax(axis=1)
    for emo, hand in zip(emo_pre, hand_pre):
        if hand == 1 and emo != 3:
            y_pred.append(2)
        else:
            y_pred.append(emo)
        y_pred_2.append(emo)
    y_pred = torch.Tensor(y_pred)
    y_pred_2 = torch.Tensor(y_pred_2)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    a = np.zeros((6,6), dtype="int")
    for i in range(len(y_true)):
        a[int(y_true[i].item())][int(y_pred[i].item())] += 1
        # if int(y_true[i].item()) != int(y_pred[i].item()):
        #     with open('./hand_fail.txt', 'a') as f:
        #         f.write(image_names[i]+'--------------'+lab2cls[int(y_pred[i].item())]+'\n')
        #         f.close()
    print(a)
    print(acc.item())
    print(f1.item())
    acc = accuracy_score(y_true, y_pred_2)
    f1 = f1_score(y_true, y_pred_2, average='macro')
    a = np.zeros((6,6), dtype="int")
    for i in range(len(y_true)):
        a[int(y_true[i].item())][int(y_pred_2[i].item())] += 1
    print(a)
    print(acc.item())
    print(f1.item())

def model_predict_hand(model, data_loader, device):
    model.eval()
    output_hand, y_true = [], []
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            out_hand = model(x_batch)
            output_hand.append(out_hand.to('cpu'))
            y_true.append(y_batch)
    y_true, output_hand = torch.cat(y_true), torch.cat(output_hand)
    hand_pred = output_hand.argmax(axis=1)
    acc = accuracy_score(y_true, hand_pred)
    f1 = f1_score(y_true, hand_pred, average='macro')
    a = np.zeros((2,2), dtype="int")
    for i in range(len(y_true)):
        a[int(y_true[i].item())][int(hand_pred[i].item())] += 1
    print(a)
    print(acc.item())
    print(f1.item())

def tsne(output_feature, y_test, save_path):
    x = np.array(output_feature)
    x_embedded = TSNE(n_components=2, init='pca').fit_transform(x)
    views_cla = [[[] for i in range(2)] for i in range(6)]

    y_test_num = len(y_test)
    for i in range(y_test_num):
        views_cla[y_test[i]][0].append(x_embedded[i][0])
        views_cla[y_test[i]][1].append(x_embedded[i][1])

    plt.figure(figsize=[10, 15], dpi=100)
    color = ['red', 'green', 'blue', 'yellow', 'pink', 'purple', 'black']
    for i in range(6):
        plt.scatter(views_cla[i][0], views_cla[i][1], c=color[i], s=50, label='class'+str(i))

    plt.legend(loc='best')
    plt.savefig(save_path, bbox_inches='tight')

def model_predict_AB(modelA, modelB, data_loader, device):
    modelA.eval(), modelB.eval()
    y_true, outputA, outputB = [], [], []

    with torch.no_grad():
        for i, (XA_batch, XB_batch, y_batch) in enumerate(data_loader):
            XA_batch, XB_batch = XA_batch.to(device), XB_batch.to(device)

            outA, outB = modelA(XA_batch), modelB(XB_batch)
            outputA.append( outA.to('cpu') )
            outputB.append( outB.to('cpu') )
            y_true.append(y_batch)

    y_true, outputA, outputB = torch.cat(y_true), torch.cat(outputA), torch.cat(outputB)
    return y_true, outputA, outputB

def model_fit(model, lr, max_epoch, data, train_option, device,save_path, print_interval=1000):
    
    opt = train_option['opt_class'](
        filter( lambda p: p.requires_grad, model.parameters() ),
        lr=lr, weight_decay=train_option['weight_decay']
    )

    best_f1 = 0.63 # 存放验证集（或测试集）最优的acc
    
    for epoch in range(max_epoch):
        train_ls = np.zeros(6)
        t0, t1 = time.time(), time.time()
        cur_lr = opt.param_groups[0]['lr']
        print('Epoch=%d, lr=%g' % (epoch, cur_lr))

        model.train()
        y_train, output_train = [], []

        for batch_i, (X_batch, y_batch, path_batch) in enumerate(data['train']):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            out = model(X_batch)
            loss = train_option['criterion'](out, y_batch)
            loss.backward()
            opt.step()

            y_train.append(y_batch.to('cpu'))
            output_train.append(out.to('cpu'))

            if batch_i % print_interval == 0:
                print('\tbatch_i=%d\tloss=%f\tin %.2f s' %(batch_i, loss.item(), time.time() - t1))
                t1 = time.time()
            for i in y_batch:
                train_ls[i.item()] += 1
        # scheduler.step()
        y_train, output_train = torch.cat(y_train), torch.cat(output_train)
        loss_train = train_option['criterion_cpu'](output_train, y_train).item()
        acc_train, f1 = metric_acc(output_train, y_train)
        print('Train\tloss=%f\tacc=%f\tf1=%f\tin %.2f s' %
               (loss_train, acc_train, f1, time.time() - t0))
        print(train_ls)                  
        if 'valid' in data:
            t0 = time.time()
            with torch.no_grad():
                y_test, output_test = model_predict1(model, data['valid'], device)
            loss_test = train_option['criterion_cpu'](output_test, y_test).item()

            acc_test, f1 = metric_acc(output_test, y_test)
            # print(classification_report(y_test, y_pred))
            ending = '\tBetter!' if f1 > best_f1 else ''
            print('valid\tloss=%f\tacc=%f\tf1=%f\tin %.2f s%s' %
                   (loss_test, acc_test, f1, time.time() - t0, ending))
            if f1 > best_f1:
                torch.save(model.state_dict(), save_path)
                best_f1 = f1

    print('best_f1 = %g' % best_f1)
    
def test(model, test_loader, device, save_path):
    model.eval()
    print('开始测试')
    output, paths = [], []
    with torch.no_grad():
        for i, (X_batch, y_batch, path_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            output = model(X_batch).to('cpu')
            # output = out
            paths = path_batch
            y_pred = output.argmax(axis=1)
            for j in range(len(y_pred)):
                with open(save_path, 'a') as f:
                    f.write(path_batch[j] +","+str(y_pred[j].item())+'\n')

def test_joint(model, model_hand, test_loader, test_hand_loader, device, save_path):
    model.eval()
    model_hand.eval()
    print('开始测试')
    output_emo, output_hand, paths = [], [], []
    with torch.no_grad():
        for i, (data1, data2) in enumerate(zip(test_loader, test_hand_loader)):
            e_x, e_y, e_p = data1
            h_x, h_y, h_p = data2
            e_x, h_x = e_x.to(device), h_x.to(device)
            output_emo = model(e_x).to('cpu')
            output_hand = model_hand(h_x).to('cpu')
            hand_pred = output_hand.argmax(axis=1)
            emo_pred = output_emo.argmax(axis=1)
            for j in range(len(hand_pred)):
                if hand_pred[j].item() == 1 and emo_pred[j].item() != 3:
                    result = 2
                else:
                    result = emo_pred[j].item()
                with open(save_path, 'a') as f:
                    f.write(e_p[j] +","+str(result)+'\n')


