#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import sys


# In[6]:


def sigmoid(x):
    r=1/(1+np.exp(-x))
    return r


# In[7]:


def sigmoid_deriv(x):
    #r=1/(1+np.exp(-x))
    return x*(1-x)


# In[8]:


def tanh(x):
    return np.tanh(x)


# In[9]:


def tanh_deriv(x):
    return 1-x*x


# In[10]:


def relu(x):
    return np.maximum(x,0)


# In[11]:


def relu_deriv(x):
    rel=np.copy(x)
    rel[rel>0]=1
    return rel


# In[12]:


def neuralnetwork(X,Y,test,act,ltype,lrate,epochs,batchsize,layers,loss,seed):
    num_layrs=len(layers)
    w=[] 
    X=X/255
    test=test/255
    np.random.seed(seed)
    #i=layers[0]
    #j=X.shape[1]
    #l=0
    j=layers[0]
    i=X.shape[1]
    l=0
    if(act==0):
        activate=sigmoid
    elif(act==1):
        activate=tanh
    else:
        activate=relu
    for k in range(0,num_layrs):
        j=layers[l]
        temp=initialise(i+1,j)
        w+=[temp]
        i=len(w[k][0])
        l=l+1
    #setseed(seed)
    #for k in range(0,num_layrs):
       # w+=[initialise(seed,i,j)]
        #i=len(w[k][0])
        #l=l+1
        #j=layers[l]
    #A_f,Z_f=forwardprop(X,w,activate,layers,loss)
    w=mini_batch_grad_descent(X,Y,w,activate,act,ltype,lrate,epochs,batchsize,layers,loss)
    A_f,Z_f=forwardprop(test,w,activate,layers,loss)
    return Z_f[num_layrs-1],w
    


# In[14]:


def mini_batch_grad_descent(X,Y,W,activate,act,ltype,lrate,epochs,batches,layers,loss):
    num=X.shape[0]//batches
    n=X.shape[0]
    lays=len(layers)
    s0=lrate
    if(act==0): #sigmoid
        deriv=sigmoid_deriv
    elif(act==1):  #tanh
        deriv=tanh_deriv 
    else:  #relu
        deriv=relu_deriv
    if(ltype==0): #fixed
        for j in range(0,epochs):
            print(j)
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                A,Z=forwardprop(mini_train,W,activate,layers,loss)
                delta=backprop(A,Z,W,mini_out,deriv,loss)
                mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
                D=np.dot(np.transpose(mini_train),delta[0])
                W[0]=W[0]-s0*D
                for k in range(1,lays):
                    D=np.dot(np.transpose(Z[k-1]),delta[k])
                    W[k]=W[k]-s0*D
                
    else: #adaptive
        for j in range(0,epochs):
            step=s0/math.sqrt(j+1)
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                A,Z=forwardprop(mini_train,W,activate,layers,loss)
                delta=backprop(A,Z,W,mini_out,deriv,loss)
                mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
                D=np.dot(np.transpose(mini_train),delta[0])
                W[0]=W[0]-step*D
                for k in range(1,lays):
                    D=np.dot(np.transpose(Z[k-1]),delta[k])
                    W[k]=W[k]-step*D
    return W     


# In[15]:


def forwardprop(X,W,activate,layers,loss):
    layrs=len(layers)
    A=[]
    Z=[]
    curr=np.c_[np.ones(X.shape[0]),X]
    for i in range(0,layrs-1):
        A+=[np.dot(curr,W[i])]
        Z+=[activate(A[i])]
        Z[i]=np.c_[np.ones((Z[i].shape[0])),Z[i]]
        curr=Z[i]
    #curr=np.c_[np.ones((curr.shape[0])),curr]
    A+=[np.dot(curr,W[layrs-1])]
    if(loss==0):
        Z+=[softmax(A[layrs-1])] 
    else:
        Z+=[activate(A[layrs-1])]
    return A,Z
    


# In[16]:


def backprop(A,Z,W,Y,deriv,loss):
    L=len(A)
    delta=[None] * L
    if(loss==0):
        delta[L-1]=(Z[L-1]-Y)/(Y.shape[0])  #for cross entropy
    else:
        delta[L-1]=np.multiply(Z[L-1]-Y,deriv(Z[L-1]))/Y.shape[0]  #MSE
    i=L-2
    while(i>=0):
        delta[i]=np.multiply(np.dot(delta[i+1],np.transpose(W[i+1])),deriv(Z[i]))
        delta[i]=delta[i][:,1:]
        i=i-1
    return delta


# In[17]:


def softmax(X):
    m=np.max(X,axis=1).reshape(-1,1)        
    e=np.exp(X-m)
    s=np.sum(e,axis=1).reshape(-1,1)
    return e/s


# In[18]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols)))


# In[ ]:


#driver
#neurala.py inputpath outputpath param.txt
inputpath=sys.argv[1]
outputpath=sys.argv[2]
train_path=inputpath+"train_data_shuffled.csv"
test_path=inputpath+"public_test.csv"
train=pd.read_csv(train_path,header=None)
test=pd.read_csv(test_path,header=None)
with open(sys.argv[3]) as sample:
    lines = sample.readlines()
params=[] #
params.append(int(lines[0].strip())) #epochs 0
params.append(int(lines[1].strip()))  #batchsize 1
params.append(int(lines[3].strip()))    #learning rate 2
params.append(float(lines[4].strip()))  #lvalue 3
params.append(int(lines[5].strip()))   #act 4
params.append(int(lines[6].strip())) #loss 5
params.append(int(lines[7].strip()))  #seed 6
y_train=np.array(train[1024])
y_train=pd.get_dummies(y_train,columns=y_train)
del test[1024]
del train[1024]
Y=y_train.to_numpy()
X=train.to_numpy()
test=test.to_numpy()
params.append(list(int(i) for i in lines[2].strip()[1:-1].split(",")))
output,W= neuralnetwork(X,Y,test,params[4],params[2],params[3],params[0],params[1],params[7],params[5],params[6])
for i in range(0,len(W)):
    str1="w_"+str(i+1)+".npy"
    #save('data.npy', data)
    np.save(outputpath+str1,W[i])
y_pred=np.argmax(output,axis=1)
np.save(outputpath+"predictions.npy",y_pred)

#epochs, batch size, a list specifying the architecture([100,50,10]implies 2 hidden layers with 100 and 
#50 neurons and 10 neurons in the output layer), learning ratetype(0 for fixed and 1 for adaptive), 
#learning rate value, activation function(0 for log sigmoid, 1 fortanh,  2 for relu),  
#loss function(0 for CE and 1 for MSE), seed value for the numpy.random.normalused(some whole number).

