import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import csv

import torch
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F
import pandas as pd
import time
import pickle
import math
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.nn import Parameter
import random

class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x) #对输入进行卷积处理，这一步output的形状是[batch,out_channels,p_w,p_h]
        outputs = outputs.view(x.size(0), -1, self.dim_caps) #将4D的卷积输出变为3D的胶囊输出形式，output的形状为[batch,caps_num,dim_caps]，其中caps_num为胶囊数量
        return squash(outputs) #激活函数，并返回激活后的胶囊

def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True) #计算输入胶囊的长度
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8) #计算缩放因子
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1) #将数据维度从[batch, in_num_caps, in_dim_caps]扩展到[batch, 1,in_num_caps, in_dim_caps,1]
        #将weight和扩展后的输入相乘，weight的尺寸是[out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]，相乘后结果尺寸为[batch, out_num_caps, in_num_caps,out_dim_caps, 1]
        #去除多余的维度，去除后结果尺寸
        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ basic ResNet class: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
    def __init__(self, block, layers, num_classes):
        
        self.inplanes = 1

        super(ResNet, self).__init__()
        
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
    
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.primarycaps = PrimaryCapsule(512, 512, 8, kernel_size=3, stride=2, padding=1)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=64*8*1, in_dim_caps=8,
                                      out_num_caps=2, out_dim_caps=16, routings=3)
       
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
       
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        # x = x.squeeze()
        # # x = x.permute(0,2,1)
        # # x = self.linear1(x)
        # # x = x.permute(0,2,1)
        # x = self.trans(x)
        # x = x.unsqueeze(1)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        
        # x = self.maxpool(x)
        # print(x.size())

        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        # print(x.shape)
        length = x.norm(dim=-1)
        # x = self.avgpool(x).view(x.size()[0], -1)
        return length
        # return x

def caps_loss(y_true, y_pred):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
         0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    return L_margin

def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 2).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
        y_pred = model(x)
        test_loss += caps_loss(y, y_pred).item() * x.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct.data.item() / len(test_loader.dataset)

def train(model, train_loader, test_loader, args,fold):
    print('Begin Training' + '-' * 70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/' + 'log_fold'+str(fold)+'.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc', 'time'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            y = torch.zeros(y.size(0), 2).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred = model(x)  # forward
            loss = caps_loss(y, y_pred)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables向后计算所有变量的所有损耗梯度
            training_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model_fold%s.h5\'' % (args.save_dir,curr_fold))
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model

def dreamer_load(sub,dimention):
    
    dataset_suffix = "_rnn_dataset.pkl"
    label_suffix = "_labels.pkl"

    dataset_dir = '/media/data/wangjinqin/EEG/Dreamer/TIME/dreamer_shuffled_data/' + "yes" + "_" + "dominance"  + "/"

    ###load training set
    with open(dataset_dir + sub + dataset_suffix, "rb") as fp:
        datasets = pickle.load(fp)
    with open(dataset_dir + sub + '_' + dimention + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        # labels = np.transpose(labels)

    labels = labels > 3
    # labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    datasets = datasets[index]  # .transpose(0,2,1)
    labels = labels[index]

    datasets = datasets.reshape(-1, 128, 14, 1).astype('float32')

    datasets = np.transpose(datasets.reshape(-1, 128, 14, 1), (0, 3, 1, 2))


    return datasets , labels

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

time_start_whole = time.time()

dataset_name = 'dreamer' #'deap' # dreamer
# subjects = ['1']
# subjects = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']  #  ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16']#,'s05']#,'s06','s07','s08']#,'s09','s10','s11','s12','s13','s14','s15','s16'，'s17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s27','s28',]
subjects = ['3'] 
dimentions = ['dominance']#,'arousal','dominance']
debaseline = 'yes' # yes or not

if __name__ == "__main__":
    seed_torch()
    for dimention in dimentions:
        for subject in subjects:
            # setting the hyper parameters
            import argparse
            import os

            parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
            parser.add_argument('--epochs', default=30, type=int)
            parser.add_argument('--batch_size', default=100, type=int)
            parser.add_argument('--lr', default=0.0001, type=float,
                                help="Initial learning rate")
            parser.add_argument('--lr_decay', default=1.0, type=float,
                                help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
            parser.add_argument('--lam_recon', default=0., type=float,
                                help="The coefficient for the loss of decoder")
            parser.add_argument('-r', '--routings', default=3, type=int,
                                help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
            parser.add_argument('--shift_pixels', default=2, type=int,
                                help="Number of pixels to shift at most in each direction.")
            parser.add_argument('--data_dir', default='./data',
                                help="Directory of data. If no data, use \'--download\' flag to download it")
            parser.add_argument('--download', action='store_true',
                                help="Download the required data.")
            parser.add_argument('--save_dir', default='./result')
            parser.add_argument('-t', '--testing', action='store_true',
                                help="Test the trained model on testing dataset")
            parser.add_argument('-w', '--weights', default=None,
                                help="The path of the saved weights. Should be specified when testing")
            args = parser.parse_args()

            print(time.asctime(time.localtime(time.time())))
            print(args)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            dataset_name == 'dreamer'  # load dreamer data
            datasets, labels = dreamer_load(subject,dimention)

            args.save_dir = args.save_dir + '/' + debaseline + '/' + subject + '_' + dimention + str(args.epochs)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            fold = 10
            test_accuracy_allfold = np.zeros(shape=[0], dtype=float)
            train_used_time_allfold = np.zeros(shape=[0], dtype=float)
            test_used_time_allfold = np.zeros(shape=[0], dtype=float)
            for curr_fold in range(fold):
                fold_size = labels.shape[0] // fold
                indexes_list = [i for i in range(len(labels))]
                # indexes = np.array(indexes_list)
                split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
                index_test = np.array(split_list)
                index_train = np.array(list(set(indexes_list) ^ set(index_test)))

                x_train = torch.from_numpy(datasets[index_train]).type(torch.FloatTensor)
                y_train = torch.from_numpy(labels[index_train]).type(torch.LongTensor)
                x_test = torch.from_numpy(datasets[index_test]).type(torch.FloatTensor)
                y_test = torch.from_numpy(labels[index_test]).type(torch.LongTensor)
                train_dataset = TensorDataset(x_train, y_train)
                train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

                test_dataset = TensorDataset(x_test, y_test)
                test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)


                model = ResNet(SEBasicBlock, [1,1,1,1], num_classes=2).cuda()
                model.cuda()
                criterion = nn.CrossEntropyLoss().cuda()
                print(model)

                # train
                train_start_time = time.time()
                train_used_time_fold = time.time() - train_start_time

                train(model, train_loader, test_loader, args,fold=curr_fold)
                print('\n Start Testing' + '-' * 70)
                test_start_time = time.time()
                test_used_time_fold = time.time() - test_start_time
                test_a_loss, test_a_acc = test(model=model, test_loader=test_loader, args=args)
                print('test_a_acc = %.4f, test_a_loss = %.5f' % (test_a_acc, test_a_loss))

                test_accuracy_allfold = np.append(test_accuracy_allfold, test_a_acc)
                train_used_time_allfold = np.append(train_used_time_allfold, train_used_time_fold)
                test_used_time_allfold = np.append(test_used_time_allfold, test_used_time_fold)
            summary = pd.DataFrame({'fold': range(1, fold + 1), 'Test accuracy': test_accuracy_allfold,'train time': train_used_time_allfold, 'test time': test_used_time_allfold})
            hyperparam = pd.DataFrame({'average acc of 10 folds': np.mean(test_accuracy_allfold),'average train time of 10 folds': np.mean(train_used_time_allfold),'average test time of 10 folds': np.mean(test_used_time_allfold),'epochs': args.epochs, 'lr': args.lr, 'batch size': args.batch_size},index=['dimention/sub'])
            writer = pd.ExcelWriter(args.save_dir + '/' + 'summary' + '_' + subject + '.xlsx')
            summary.to_excel(writer, 'Result', index=False)
            hyperparam.to_excel(writer, 'HyperParam', index=False)
            writer.save()
            print('10 fold average accuracy: ', np.mean(test_accuracy_allfold))
            print('10 fold average train time: ', np.mean(train_used_time_allfold))
            print('10 fold average test time: ', np.mean(test_used_time_allfold))
                
