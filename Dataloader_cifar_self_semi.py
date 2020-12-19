from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import json
import os
import torch
from torchnet.meter import AUCMeter
from itertools import chain

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test' or self.mode=='test_average':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        
        else:
            train_data = []
            train_label = []
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if self.mode == 'eval' or self.mode == 'eval_average':
                self.eval_data = train_data[45000:]
                self.eval_label = train_label[45000:]

            else:
                if os.path.exists(noise_file):
                    noise_label = json.load(open(noise_file,"r"))
                else:    #inject noise   
                    noise_label = []
                    if self.mode in ['all', 'benchmark_all', 'benchmark_all_average']:
                      size = 50000
                    elif self.mode in ['train', 'benchmark', 'benchmark_average']:
                      size = 45000
                    idx = list(range(size))
                    random.shuffle(idx)
                    num_noise = int(self.r*size)            
                    noise_idx = idx[:num_noise]
                    for i in range(size):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)
                            elif noise_mode=='asym':   
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)                    
                        else:    
                            noise_label.append(train_label[i])   
                    print("save noisy labels to %s ..."%noise_file)        
                    json.dump(noise_label,open(noise_file,"w"))       

                if self.mode in ['all', 'benchmark_all', 'benchmark_all_average']:
                    self.train_data = train_data
                    self.noise_label = noise_label
                    self.clean_label = train_label
            
                elif self.mode in ['train', 'benchmark', 'benchmark_average']:
                    self.train_data = train_data[:45000]
                    self.noise_label = noise_label[:45000]
                    self.clean_label = train_label[:45000]
                    
                else:                   
                    if self.mode == "labeled":
                        pred_idx = pred.nonzero()[0]
                        self.probability = [probability[i] for i in pred_idx]
                    
                        clean = (np.array(noise_label)==np.array(train_label))                                                       
                        auc_meter = AUCMeter()
                        auc_meter.reset()
                        auc_meter.add(probability,clean)        
                        auc,_,_ = auc_meter.value()               
                        log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                        log.flush()      
                    
                    elif self.mode == "unlabeled":
                        pred_idx = (1-pred).nonzero()[0]                                             
                
                    self.train_data = train_data[pred_idx]
                    self.noise_label = [noise_label[i] for i in pred_idx]                          
                    print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                

    def __getitem__(self, index):

        if self.mode == 'eval_average':
            img, target = self.eval_data[index], self.eval_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            angle = [0, 90, 180, 270]
            img1 = rotate(img, 90, axes=(1,2), reshape=False)
            img2 = rotate(img, 180, axes=(1,2), reshape=False)
            img3 = rotate(img, 270, axes=(1,2), reshape=False)
            return img, img1, img2, img3, target, index

        elif self.mode=='test_average':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            angle = [0, 90, 180, 270]
            img1 = rotate(img, 90, axes=(1,2), reshape=False)
            img2 = rotate(img, 180, axes=(1,2), reshape=False)
            img3 = rotate(img, 270, axes=(1,2), reshape=False)
            return img, img1, img2, img3, target, index

        elif self.mode=='eval':
            img, target = self.eval_data[index], self.eval_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index

        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        
        elif self.mode=='benchmark_average' or self.mode=='benchmark_all_average':
            img, target, target_clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            angle = [0, 90, 180, 270]
            img1 = rotate(img, 90, axes=(1,2), reshape=False)
            img2 = rotate(img, 180, axes=(1,2), reshape=False)
            img3 = rotate(img, 270, axes=(1,2), reshape=False)
            target_rot = [0, 1, 2, 3]
            return img, img1, img2, img3, target, target_rot, target_clean, index

        elif self.mode=='benchmark' or self.mode=='benchmark_all':
            img, target, target_clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            target_rot = 0
            return img, target, target_rot, target_clean, index

        else: # labeled, unlabeled, all, train
            img, target, target_clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            angle = [0, 90, 180, 270]
            target_rot = np.random.choice([0, 1, 2, 3])
            img = rotate(img, angle[target_rot], axes=(1,2), reshape=False)
            return img, target, target_rot, target_clean, index
           
    def __len__(self):
        if self.mode in ['test', 'test_average']:
            return len(self.test_data)
        elif self.mode in ['eval', 'eval_average']:
            return len(self.eval_data)
        else:
            return len(self.train_data)


class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
            
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup_all':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="benchmark_all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

        elif mode=='warmup':
            train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="benchmark",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

        elif mode=='train_all':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                 batch_size=self.batch_size*2,
                 shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        elif mode=='train':
            train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="train",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train_separate':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob, log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='eval_train_all':
            eval_train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='benchmark_all', noise_file=self.noise_file)      
            eval_train_loader = DataLoader(
                dataset=eval_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_train_loader

        elif mode=='eval_train':
            eval_train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='benchmark', noise_file=self.noise_file)      
            eval_train_loader = DataLoader(
                dataset=eval_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_train_loader

        elif mode=='eval':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='eval')      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        
        elif mode=='eval_train_all_average':
            eval_train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='benchmark_all_average', noise_file=self.noise_file)      
            eval_train_loader = DataLoader(
                dataset=eval_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_train_loader

        elif mode=='eval_train_average':
            eval_train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='benchmark_average', noise_file=self.noise_file)      
            eval_train_loader = DataLoader(
                dataset=eval_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_train_loader

        elif mode=='eval_average':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='eval_average')      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode=='test_average':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test_average')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader