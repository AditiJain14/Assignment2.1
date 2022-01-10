#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt


# In[8]:


def sigmoid(x):
    r=1/(1+np.exp(-x))
    return r


# In[9]:


def sigmoid_deriv(x):
    return x*(1-x)


# In[10]:


def tanh(x):
    return np.tanh(x)


# In[11]:


def tanh_deriv(x):
    return 1-x*x


# In[12]:


def relu(x):
    return np.maximum(0,x)


# In[13]:


def relu_deriv(x):
    x[x>0]=1
    return x


# In[39]:


def neuralnetwork(X,Y,test,act,ltype,lrate,epochs,batchsize,layers,seed):
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
    
    w=mini_batch_grad_descent(X,Y,w,activate,act,ltype,lrate,epochs,batchsize,layers)
    A_f,Z_f=forwardprop(X,w,activate,layers)
    return Z_f[num_layrs-1]
    


# In[40]:


def mini_batch_grad_descent(X,Y,W,activate,act,ltype,lrate,epochs,batches,layers):
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
    for j in range(0,epochs):
        for i in range(0,num):
            mini_train=X[i*batches:(i+1)*batches]
            mini_out=Y[i*batches:(i+1)*batches]
            A,Z=forwardprop(mini_train,W,activate,layers)
            delta=backprop(A,Z,W,mini_out,deriv)
            mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
            D=np.dot(np.transpose(mini_train),delta[0])
            W[0]=W[0]-s0*D
            for k in range(1,lays):
                D=np.dot(np.transpose(Z[k-1]),delta[k])
                W[k]=W[k]-s0*D
    return W     


# In[41]:


def forwardprop(X,W,activate,layers):
    layrs=len(layers)
    A=[]
    Z=[]
    curr=np.c_[np.ones(X.shape[0]),X]
    for i in range(0,layrs-1):
        A+=[np.dot(curr,W[i])]
        Z+=[activate(A[i])]
        Z[i]=np.c_[np.ones((Z[i].shape[0])),Z[i]]
        curr=Z[i]
    A+=[np.dot(curr,W[layrs-1])]
    Z+=[softmax(A[layrs-1])] 
    return A,Z
    


# In[42]:


def backprop(A,Z,W,Y,deriv):
    L=len(A)
    delta=[None] * L
    delta[L-1]=(Z[L-1]-Y)/(Y.shape[0])  #for cross entropy
    i=L-2
    while(i>=0):
        delta[i]=np.multiply(np.dot(delta[i+1],np.transpose(W[i+1])),deriv(Z[i]))
        delta[i]=delta[i][:,1:]
        i=i-1
    return delta


# In[43]:


def softmax(X):
    m=np.max(X,axis=1).reshape(-1,1)        
    e=np.exp(X-m)
    s=np.sum(e,axis=1).reshape(-1,1)
    return e/s


# In[44]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols)))


# In[ ]:


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
del test[1024]
del train[1024]
Y=y_train.to_numpy()
X=train.to_numpy()
test=test.to_numpy()


# In[46]:


loss1_s=[]
loss1_b=[]
epochs=[1,3,5,10,15,20]
for i in epochs :
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[512,256,128,64,46],1)
    loss1_b+=[cross_entropy(Y,output)]
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[256,46],1)
    loss1_s+=[cross_entropy(Y,output)]


# In[ ]:


# 0.19 sigmoid 15 epochs with standard


# In[ ]:


#0.835 tanh 15 epochs with standard


# In[ ]:


#0.7254 relu 15 epochs with standard


# In[ ]:


#0.6993 relu 15 epochs with adaptive
#0.7513 tanh 15 epochs with adaptive
#


# # Moment Gradient Descent

# In[58]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols))), np.zeros((rows,cols))


# In[59]:


def neuralnetwork(X,Y,test,act,ltype,lrate,epochs,batchsize,layers,seed):
    num_layrs=len(layers)
    w=[] 
    v=[]
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
        temp,v_t=initialise(i+1,j)
        w+=[temp]
        v+=[v_t]
        i=len(w[k][0])
        l=l+1
    #setseed(seed)
    #for k in range(0,num_layrs):
       # w+=[initialise(seed,i,j)]
        #i=len(w[k][0])
        #l=l+1
        #j=layers[l]
    w=mini_batch_grad_descent(X,Y,w,v,activate,act,ltype,lrate,epochs,batchsize,layers)
    A_t,Z_t=forwardprop(X,w,activate,layers)
    #A_f,Z_f=forwardprop(test,w,activate,layers)
    return Z_t[num_layrs-1]
    


# In[62]:


def mini_batch_grad_descent(X,Y,W,v,activate,act,ltype,lrate,epochs,batches,layers):
    num=X.shape[0]//batches
    n=X.shape[0]
    lays=len(layers)
    s0=lrate
    frac=0.9
    if(act==0): #sigmoid
        deriv=sigmoid_deriv
    elif(act==1):  #tanh
        deriv=tanh_deriv 
    else:  #relu
        deriv=relu_deriv
    if(ltype==0): #fixed
        for j in range(0,epochs):
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                A,Z=forwardprop(mini_train,W,activate,layers)
                delta=backprop(A,Z,W,mini_out,deriv)
                mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
                D=np.dot(np.transpose(mini_train),delta[0])
                v[0]=frac*v[0]+s0*D
                W[0]=W[0]-v[0]
                for k in range(1,lays):
                    D=np.dot(np.transpose(Z[k-1]),delta[k])
                    v[k]=frac*v[k]+s0*D
                    W[k]=W[k]-v[k]
                
    else: #adaptive
        for j in range(0,epochs):
            step=s0/math.sqrt(j+1)
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                A,Z=forwardprop(mini_train,W,activate,layers)
                delta=backprop(A,Z,W,mini_out,deriv)
                mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
                D=np.dot(np.transpose(mini_train),delta[0])
                v[0]=frac*v[0]-step*D
                W[0]=W[0]-v[0]
                for k in range(1,lays):
                    D=np.dot(np.transpose(Z[k-1]),delta[k])
                    v[k]=frac*v[k]-step*D
                    W[k]=W[k]-v[k]
    return W     


# In[64]:


loss2_s=[]
loss2_b=[]
epochs=[1,3,5,10,15,20]
for i in epochs :
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[512,256,128,64,46],1)
    loss2_b+=[cross_entropy(Y,output)]
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[256,46],1)
    loss2_s+=[cross_entropy(Y,output)]


# # Nesterov accelerated gradient

# In[65]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols))), np.zeros((rows,cols))


# In[66]:


def neuralnetwork(X,Y,test,act,ltype,lrate,epochs,batchsize,layers,seed):
    num_layrs=len(layers)
    w=[] 
    v=[]
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
        temp,v_t=initialise(i+1,j)
        w+=[temp]
        v+=[v_t]
        i=len(w[k][0])
        l=l+1
    #setseed(seed)
    #for k in range(0,num_layrs):
       # w+=[initialise(seed,i,j)]
        #i=len(w[k][0])
        #l=l+1
        #j=layers[l]
    w=mini_batch_grad_descent(X,Y,w,v,activate,act,ltype,lrate,epochs,batchsize,layers)
    #A_f,Z_f=forwardprop(test,w,activate,layers)
    A_t,Z_t=forwardprop(X,w,activate,layers)
    return Z_t[num_layrs-1]
        


# In[69]:


def mini_batch_grad_descent(X,Y,W,v,activate,act,ltype,lrate,epochs,batches,layers):
    num=X.shape[0]//batches
    n=X.shape[0]
    lays=len(layers)
    s0=lrate
    frac=0.9
    if(act==0): #sigmoid
        deriv=sigmoid_deriv
    elif(act==1):  #tanh
        deriv=tanh_deriv 
    else:  #relu
        deriv=relu_deriv
    if(ltype==0): #fixed
        for j in range(0,epochs):
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                W_=[]
                for i in range(0,len(W)):
                    W_+=[W[i]-frac*v[i]]
                A,Z=forwardprop(mini_train,W_,activate,layers)
                delta=backprop(A,Z,W,mini_out,deriv)
                mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
                D=np.dot(np.transpose(mini_train),delta[0])
                v[0]=frac*v[0]+s0*D
                W[0]=W[0]-v[0]
                for k in range(1,lays):
                    D=np.dot(np.transpose(Z[k-1]),delta[k])
                    v[k]=frac*v[k]+s0*D
                    W[k]=W[k]-v[k]
                
    else: #adaptive
        for j in range(0,epochs):
            step=s0/math.sqrt(j+1)
            for i in range(0,num):
                mini_train=X[i*batches:(i+1)*batches]
                mini_out=Y[i*batches:(i+1)*batches]
                A,Z=forwardprop(mini_train,W-frac*v,activate,layers)
                delta=backprop(A,Z,W,mini_out,deriv)
                mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
                D=np.dot(np.transpose(mini_train),delta[0])
                v[0]=frac*v[0]-step*D
                W[0]=W[0]-v[0]
                for k in range(1,lays):
                    D=np.dot(np.transpose(Z[k-1]),delta[k])
                    v[k]=frac*v[k]-step*D
                    W[k]=W[k]-v[k]
                
    return W     


# In[71]:


loss3_s=[]
loss3_b=[]
epochs=[1,3,5,10,15,20]
for i in epochs :
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[512,256,128,64,46],1)
    loss3_b+=[cross_entropy(Y,output)]
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[256,46],1)
    loss3_s+=[cross_entropy(Y,output)]


# # Adam 

# In[80]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols))), np.zeros((rows,cols)),np.zeros((rows,cols))


# In[81]:


def neuralnetwork(X,Y,test,act,ltype,lrate,epochs,batchsize,layers,seed):
    num_layrs=len(layers)
    w=[] 
    m=[]
    v=[]
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
        temp,m_t,v_t=initialise(i+1,j)
        w+=[temp]
        m+=[m_t]
        v+=[v_t]
        i=len(w[k][0])
        l=l+1
    #setseed(seed)
    #for k in range(0,num_layrs):
       # w+=[initialise(seed,i,j)]
        #i=len(w[k][0])
        #l=l+1
        #j=layers[l]
    w=mini_batch_grad_descent(X,Y,w,m,v,activate,act,ltype,lrate,epochs,batchsize,layers)
    #A_f,Z_f=forwardprop(test,w,activate,layers)
    A_t,Z_t=forwardprop(X,w,activate,layers)
    return Z_t[num_layrs-1]
    


# In[82]:


def mini_batch_grad_descent(X,Y,W,m,v,activate,act,ltype,lrate,epochs,batches,layers):
    num=X.shape[0]//batches
    n=X.shape[0]
    lays=len(layers)
    s0=lrate
    b1=0.9
    b2=0.999
    e=0.00000001
    t=0
    if(act==0): #sigmoid
        deriv=sigmoid_deriv
    elif(act==1):  #tanh
        deriv=tanh_deriv 
    else:  #relu
        deriv=relu_deriv
    for j in range(0,epochs):
        for i in range(0,num):
            t=t+1
            mini_train=X[i*batches:(i+1)*batches]
            mini_out=Y[i*batches:(i+1)*batches]
            A,Z=forwardprop(mini_train,W,activate,layers)
            delta=backprop(A,Z,W,mini_out,deriv)
            mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
            D=np.dot(np.transpose(mini_train),delta[0])
            m[0]=b1*m[0]+(1-b1)*D
            v[0]=b2*v[0]+(1-b2)*np.multiply(D,D)
            m_t=m[0]/(1-(b1**t))
            v_t=v[0]/(1-(b2**t))
            W[0]=W[0]-(s0*m_t)/(np.sqrt(v_t)+e)
            for k in range(1,lays):
                D=np.dot(np.transpose(Z[k-1]),delta[k])
                m[k]=b1*m[k]+(1-b1)*D
                v[k]=b2*v[k]+(1-b2)*np.multiply(D,D)
                m_t=m[k]/(1-(b1**t))
                v_t=v[k]/(1-(b2**t))
                W[k]=W[k]-(s0*m_t)/(np.sqrt(v_t)+e)
                            
    return W     


# In[84]:


loss4_s=[]
loss4_b=[]
epochs=[1,3,5,10,15,20]
for i in epochs :
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[512,256,128,64,46],1)
    loss4_b+=[cross_entropy(Y,output)]
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[256,46],1)
    loss4_s+=[cross_entropy(Y,output)]


# In[ ]:





# # RMSProp

# In[87]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols))),np.zeros((rows,cols))


# In[88]:


def neuralnetwork(X,Y,test,act,ltype,lrate,epochs,batchsize,layers,seed):
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
    w=mini_batch_grad_descent(X,Y,w,e,activate,act,ltype,lrate,epochs,batchsize,layers)
    #A_f,Z_f=forwardprop(test,w,activate,layers)
    A_t,Z_t=forwardprop(X,w,activate,layers)
    return Z_t[num_layrs-1]
    


# In[89]:


def mini_batch_grad_descent(X,Y,W,e,activate,act,ltype,lrate,epochs,batches,layers):
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
            A,Z=forwardprop(mini_train,W,activate,layers)
            delta=backprop(A,Z,W,mini_out,deriv)
            mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
            D=np.dot(np.transpose(mini_train),delta[0])
            e[0]=0.9*e[0]+0.1*np.multiply(D,D)
            W[0]=W[0]-(s0*D)/(np.sqrt(e[0]+e1))
            for k in range(1,lays):
                D=np.dot(np.transpose(Z[k-1]),delta[k])
                e[k]=0.9*e[k]+0.1*np.multiply(D,D)
                W[k]=W[k]-(s0*D)/(np.sqrt(e[k]+e1))
                            
    return W     


# In[90]:


loss5_s=[]
loss5_b=[]
epochs=[1,3,5,10,15,20]
for i in epochs :
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[512,256,128,64,46],1)
    loss5_b+=[cross_entropy(Y,output)]
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[256,46],1)
    loss5_s+=[cross_entropy(Y,output)]


# # Nadam

# In[91]:


def initialise(rows,cols):
    return np.float32(np.random.normal(size=(rows, cols))*math.sqrt(2/(rows+cols))), np.zeros((rows,cols)),np.zeros((rows,cols))


# In[92]:


def neuralnetwork(X,Y,test,act,ltype,lrate,epochs,batchsize,layers,seed):
    num_layrs=len(layers)
    w=[] 
    m=[]
    v=[]
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
        temp,m_t,v_t=initialise(i+1,j)
        w+=[temp]
        m+=[m_t]
        v+=[v_t]
        i=len(w[k][0])
        l=l+1
    #setseed(seed)
    #for k in range(0,num_layrs):
       # w+=[initialise(seed,i,j)]
        #i=len(w[k][0])
        #l=l+1
        #j=layers[l]
    w=mini_batch_grad_descent(X,Y,w,m,v,activate,act,ltype,lrate,epochs,batchsize,layers)
    #A_f,Z_f=forwardprop(test,w,activate,layers)
    A_t,Z_t=forwardprop(X,w,activate,layers)
    return Z_t[num_layrs-1]
    


# In[93]:


def mini_batch_grad_descent(X,Y,W,m,v,activate,act,ltype,lrate,epochs,batches,layers):
    num=X.shape[0]//batches
    n=X.shape[0]
    lays=len(layers)
    s0=lrate
    b1=0.9
    b2=0.999
    e=0.00000001
    t=0
    if(act==0): #sigmoid
        deriv=sigmoid_deriv
    elif(act==1):  #tanh
        deriv=tanh_deriv 
    else:  #relu
        deriv=relu_deriv
    for j in range(0,epochs):
        for i in range(0,num):
            t=t+1
            mini_train=X[i*batches:(i+1)*batches]
            mini_out=Y[i*batches:(i+1)*batches]
            A,Z=forwardprop(mini_train,W,activate,layers)
            delta=backprop(A,Z,W,mini_out,deriv)
            mini_train=np.c_[np.ones(mini_train.shape[0]),mini_train]
            D=np.dot(np.transpose(mini_train),delta[0])
            m[0]=b1*m[0]+(1-b1)*D
            v[0]=b2*v[0]+(1-b2)*np.multiply(D,D)
            m_t=m[0]/(1-(b1**t))
            v_t=v[0]/(1-(b2**t))
            W[0]=W[0]-(s0*(b1*m_t+((1-b1)*D/(1-b1**t))))/(np.sqrt(v_t)+e)
            for k in range(1,lays):
                D=np.dot(np.transpose(Z[k-1]),delta[k])
                m[k]=b1*m[k]+(1-b1)*D
                v[k]=b2*v[k]+(1-b2)*np.multiply(D,D)
                m_t=m[k]/(1-(b1**t))
                v_t=v[k]/(1-(b2**t))
                W[k]=W[k]-(s0*(b1*m_t+((1-b1)*D/(1-b1**t))))/(np.sqrt(v_t)+e)
                            
    return W     


# In[95]:


loss6_s=[]
loss6_b=[]
epochs=[1,3,5,10,15,20]
for i in epochs :
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[512,256,128,64,46],1)
    loss6_b+=[cross_entropy(Y,output)]
    output=neuralnetwork(X,Y,test,1,0,0.01,i,100,[256,46],1)
    loss6_s+=[cross_entropy(Y,output)]


# In[57]:


plt.figure(figsize=(11, 8))


plt.plot(epochs, loss1_s,label='Standard')
plt.plot(epochs, loss2_s, color='r',label='Momentum')
plt.plot(epochs, loss3_s, color='g',label='Nesterov')
plt.plot(epochs, loss4_s, color='b',label='Adam')
plt.plot(epochs, loss5_s, label='RmsProp')
plt.plot(epochs, loss6_s, color='y',label='Nadam')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.title("Optimizers with 2 layers")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
#plt.show()
plt.savefig(outputpath+"c1_small.png")


# loss2=[2.210795078918117, 1.4546499670073332, 1.1682186055850796, 0.7575743882811939, 0.5112178872050921]
# loss3=[2.2101162755129633, 1.4541123416065007, 1.1676567776293332, 0.7568065964586597, 0.5100973391155798]
# loss4=[0.4749634001632073, 0.19405171782492947, 0.14314000707152327, 0.08434321072415789, 0.055290691613986385]
# loss5=[0.5172509697400426, 0.19222627394674033, 0.10655289714680836, 0.06571545607203538, 0.056894155841372265]
# loss6=[0.4260623585766086, 0.16439733828685724, 0.12797705801365084, 0.10236318449868731, 0.07024000258323351]

# In[97]:


plt.figure(figsize=(11, 8))


plt.plot(epochs, loss1_b,label='Standard')
plt.plot(epochs, loss2_b, color='r',label='Momentum')
plt.plot(epochs, loss3_b, color='g',label='Nesterov')
plt.plot(epochs, loss4_b, color='b',label='Adam')
plt.plot(epochs, loss5_b, label='RmsProp')
plt.plot(epochs, loss6_b, color='y',label='Nadam')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.title("Optimizers with 5 layers")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
#plt.show()
plt.savefig(outputpath+"c1_big.png")


# In[ ]:




