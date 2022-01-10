#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import sys
import time
init=time.time()
import matplotlib.pyplot as plt

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

def neuralnetwork(X,Y,test,act,lrate,epochs,batchsize,layers,loss,seed):
    num_layrs=len(layers)
    w=[] 
    e=[]
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
        temp,e_=initialise(i+1,j)
        w+=[temp]
        e+=[e_]
        i=len(w[k][0])
        l=l+1
    #setseed(seed)
    #for k in range(0,num_layrs):
       # w+=[initialise(seed,i,j)]
        #i=len(w[k][0])
        #l=l+1
        #j=layers[l]
    w=mini_batch_grad_descent(X,Y,w,e,activate,act,lrate,epochs,batchsize,layers,loss)
    A_f,Z_f=forwardprop(test,w,activate,layers,loss)
    #A_t,Z_t=forwardprop(X,w,activate,layers)
    return w,Z_f[num_layrs-1]
        


# In[2]:


def mini_batch_grad_descent(X,Y,W,e,activate,act,lrate,epochs,batches,layers,loss):
    num=X.shape[0]//batches
    n=X.shape[0]
    lays=len(layers)
    s0=lrate
    e1=0.00000001
    if(act==0): #sigmoid
        deriv=sigmoid_deriv
    elif(act==1):  #tanh
        deriv=tanh_deriv 
    else:  #relu
        deriv=relu_deriv
    for j in range(0,epochs):
        for i in range(0,num):
            mini_train=X[i*batches:(i+1)*batches]
            mini_out=Y[i*batches:(i+1)*batches]
            A,Z=forwardprop(mini_train,W,activate,layers,loss)
            delta=backprop(A,Z,W,mini_out,deriv,loss)
            mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
            D=np.dot(np.transpose(mini_train),delta[0])
            e[0]=0.9*e[0]+0.1*np.multiply(D,D)
            W[0]=W[0]-(s0*D)/(np.sqrt(e[0]+e1))
            for k in range(1,lays):
                D=np.dot(np.transpose(Z[k-1]),delta[k])
                e[k]=0.9*e[k]+0.1*np.multiply(D,D)
                W[k]=W[k]-(s0*D)/(np.sqrt(e[k]+e1))
                            
    return W     

# In[3]:


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
    


# In[4]:


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


# In[5]:


def softmax(X):
    m=np.max(X,axis=1).reshape(-1,1)        
    e=np.exp(X-m)
    s=np.sum(e,axis=1).reshape(-1,1)
    return e/s


# In[6]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols))),np.zeros((rows,cols))

def cross_entropy(Y_test,output):
    output[output==0]=1e-20
    return (-1/Y_test.shape[0])*np.sum(np.multiply(Y_test,np.log(output)))


# In[ ]:


inputpath=sys.argv[1]
outputpath=sys.argv[2]
train_path=inputpath+"train_data_shuffled.csv"
test_path=inputpath+"public_test.csv"
train=pd.read_csv(train_path,header=None)
test=pd.read_csv(test_path,header=None)
y_train=np.array(train[1024])
y_train=pd.get_dummies(y_train,columns=y_train)
y_test=np.array(test[1024])
y_test=pd.get_dummies(y_test,columns=y_test)
del test[1024]
del train[1024]
Y=y_train.to_numpy()
X=train.to_numpy()
test=test.to_numpy()
y_test=y_test.to_numpy()


# In[ ]:


arch=[[256,46],[256,128,46],[128,64,46],[512,46],[400,46]]
#epochs=[1,3,5,10,15,20]
epochs=[1,3,5]
k=0
for layer in arch:
    loss=[]
    k=k+1
    for i in epochs:
        w,output=neuralnetwork(X,Y,test,1,0.001,i,100,layer,0,1)
        loss+=[cross_entropy(y_test,output)]
    plt.figure(figsize=(11, 8))
    plt.plot(epochs,loss,label=str(layer))
  
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title(str(layer))
    name="d1"+str(k)+".png"
    plt.savefig(outputpath+name)

