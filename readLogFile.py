# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:56:09 2018

@author: mor1joseph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_fc1 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_fc2\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_fc2 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_fc3\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_fc3 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_fc4\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_fc4 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_fc5\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_fc5 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_fc6\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)

data_nuc1 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_nuc2\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_nuc2 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_nuc3\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_nuc3 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_nuc4\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_nuc4 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_nuc5\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)
data_nuc5 = pd.read_csv(r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\NALU-pytorch-master\log_nuc6\log_train.txt',header=None,skiprows=[0],names=['0','1','2','3','4','5','6'], engine='python', delim_whitespace=True)

data = [data_fc1,data_fc2,data_fc3,data_fc4,data_fc5,data_nuc1,data_nuc2,data_nuc3,data_nuc4,data_nuc5]
acc_test_fc2 = dict()
loss_test_fc2 =dict()
loss_train_fc2 =dict()
loss_train_avg_fc2=dict()
name = ['fc1','fc2','fc3','fc4','fc5','nuc1','nuc2','nuc3','nuc4','nuc5']

ii=0
for d in data:
    nm = name[ii]
    acc_test_fc2[nm] = [float(d['6'][i][:-1]) for i in range(len(d['0'])) if str(d['0'][i]).startswith("Test")]
    loss_test_fc2[nm] = [float(d['4'][i][:-1]) for i in range(len(d['0'])) if str(d['0'][i]).startswith("Test")]
    
    loss_train_fc2[nm] = [float(d['6'][i][:-1]) for i in range(len(d['0'])) if str(d['0'][i]).startswith("Train")]
    n = 157
    loss_train_avg_fc2[nm] =[np.divide(sum(loss_train_fc2[nm][i:i+n]),157.0) for i in range(0,len(loss_train_fc2[nm]),n)]
    ii+=1


epoch = np.arange(0,100,1)

plt.subplot(5,2,1)
plt.plot(epoch,loss_train_avg_fc2['fc1'])
plt.title('FC')
plt.ylabel('loss')
plt.subplot(5,2,3)
plt.plot(epoch,loss_train_avg_fc2['fc2'])
plt.ylabel('loss')
plt.subplot(5,2,5)
plt.plot(epoch,loss_train_avg_fc2['fc3'])
plt.ylabel('loss')
plt.subplot(5,2,7)
plt.plot(epoch,loss_train_avg_fc2['fc4'])
plt.ylabel('loss')
plt.subplot(5,2,9)
plt.plot(epoch,loss_train_avg_fc2['fc5'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.subplot(5,2,2)
plt.plot(epoch,loss_train_avg_fc2['nuc1'])
plt.ylabel('loss')
plt.title('NAC')
plt.subplot(5,2,4)
plt.plot(epoch,loss_train_avg_fc2['nuc2'])
plt.ylabel('loss')
plt.subplot(5,2,6)
plt.plot(epoch,loss_train_avg_fc2['nuc3'])
plt.ylabel('loss')
plt.subplot(5,2,8)
plt.plot(epoch,loss_train_avg_fc2['nuc4'])
plt.ylabel('loss')
plt.subplot(5,2,10)
plt.plot(epoch,loss_train_avg_fc2['nuc5'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()