import numpy as np
import h5py
from random import randint
import time 

#loading MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

def softmax(z):
    array = np.exp(z - max(z))/(np.sum(np.exp(z - max(z))))
    return array

def relu(z):
    return np.maximum(z,0)

def relu_d(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z
    
def filter_initial(f_s):
    filt = np.random.randn(f_s,f_s)/np.sqrt(f_s)
    return filt

def convolution(inp,i_w,filt,n_f): #inp is input array
    f_s = filt.shape[1]    
    output = np.zeros((i_w-f_s+1,i_w-f_s+1,n_f))
    for p in range(n_f):
        for i in range(i_w-f_s+1):
            for j in range(i_w-f_s+1):
                f_v = inp[i:i+f_s,j:j+f_s] #field of vision or segment of input being multiplied
                elem_mult = np.multiply(f_v,filt[:,:,p])
                add = np.sum(elem_mult) #adding result of element wise mult
                output[i,j,p] = add
    return output
  
    
#calculating test data output
def test(X,b,W,K,i_w,n_f):
    X=np.reshape(X,(i_w,i_w))
    Z = convolution(X,i_w,K,n_f)
    H = relu(Z)
    for i in range(10):
        U[i,:] = np.sum(np.multiply(W[i,:,:,:],H))+b[i,:]
    
    Fx = softmax(U)    
    return Fx     
    

#Hyperparameters
num_epochs = 10
LR = 0.01
i_w =28 #img width
i_d = 1 #img depth
filt_size = 3 #filter size
n_f = 3 #no of filters
dim = i_w-filt_size+1

##parameters of neural network - initialization
W = np.random.randn(10,dim,dim,n_f)/np.sqrt(dim)
#W = np.random.randn(10,2028)/np.sqrt(2028)
b = np.random.randn(10,1)/np.sqrt(10)
U = np.zeros((10,1))
d = np.zeros((dim,dim,n_f))
sigma_d = np.zeros((dim,dim,n_f))
K = np.random.randn(filt_size,filt_size,n_f)/np.sqrt(filt_size)



for epoch in range(num_epochs):
    correct = 0
    start = time.time()
    for i in range(len(x_train)):
    #for i in range(2500):
        #schedule for the learning rate
        if (epoch > 5):
            LR = 0.001
        if (epoch > 10):
            LR = 0.0001
        if (epoch > 15):
            LR = 0.00001
        
        #picking a random sample from training dataset
        n_random = randint(0,len(x_train)-1)
        y = y_train[n_random]
        Y = np.zeros((10,1))
        Y[y,0]=1
        
        X = x_train[n_random][:]
        X=np.reshape(X,(i_w,i_w))
        

        Z = convolution(X,i_w,K,n_f)
        H = relu(Z)
        
        H = np.reshape(H,(2028,1))      
        W = np.reshape(W,(10,2028))
        
        U = np.matmul(W,H)+b
        
        H = np.reshape(H,(26,26,3))
        W = np.reshape(W,(10,dim,dim,n_f))
        #for i in range(10):
            #U[i,:] = np.sum(np.multiply(W[i,:,:,:],H))+b[i,:]
        
        Fx = softmax(U)
        pred = np.argmax(Fx) 
        if pred==y:
            correct+=1
        
        dU = Fx - Y
        
        for p in range(n_f):
            for i in range(dim):
                for j in range(dim):
                    d[i,j,p] = np.sum(np.multiply(dU,W[:,i,j,p]))        
        
        relu_diff = relu_d(Z)
        for p in range(n_f):             
            sigma_d[:,:,p] = np.multiply(relu_diff[:,:,p],d[:,:,p])
        
        dK = convolution(X,i_w,sigma_d,n_f)
                
        #updation step
        b = b - LR*dU
        for i in range(10):
            W[i,:,:,:] = W[i,:,:,:] - LR*dU[i,:]*H
        
        K = K - LR*dK
        
    print('Epoch:'+str(epoch)+' Training data Accuracy: {}'.format((correct/np.float(len(x_train)))*100))
    finish = time.time()
    print(finish-start)
    
    
#Running model on test data        
correct = 0    
for i in range(len(x_test)):   
    y_calc = test(x_test[i],b,W,K,i_w,n_f) #returns result of softmax
    index = np.argmax(y_calc)
    if index == y_test[i]:
        correct+=1

accuracy = (correct/np.float(len(x_test)))*100     
print('\nTest data accuracy: {}'.format(accuracy))  