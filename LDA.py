import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Extract data
data = pd.read_csv("./data/project_train.csv")
inputs = data.iloc[:,:11]
labels = data.iloc[:,11]


#L=np.array([0,1,1,1,0,1])
# L1=np.array([[0.3,0.4,0.6,0],[0.4,0.5,0.8,1],[0.1,0.9,0.8,1],[0.3,0.6,0.8,0],[0.5,0.9,0.7,1]])

#LDA
#first estimate p_Y using the empirical distribution
def p_Y(labels):
    p_Y1=(1/len(labels))*np.sum(labels)
    p_Y0=1-p_Y1
    return [p_Y0,p_Y1]
#print(p_Y(labels)[0])
#print(p_Y(labels)[1])

#estimate means within each class
#len(A)*p_Y(A)[0] represents the number of examples in class 0 in the training set
def mu_0(data):    
    #K consists of all vectors with label 0
    K=data.loc[data.iloc[:,-1]==0.]
    return (1/(len(K)))*np.sum(K.iloc[:,:-1], axis=0)
def mu_1(data):
    #K consists of all vectors with label 1
    K=data.loc[data.iloc[:,-1]==1.]
    return (1/(len(K)))*np.sum(K.iloc[:,:-1], axis=0)
print(mu_0(data))
print(mu_1(data))

def sigma(data):
    K0=data.loc[data.iloc[:,-1]==0.]
    K1=data.loc[data.iloc[:,-1]==1.]
    #we subtract the mean from every row
    K0=K0.iloc[:,:-1].subtract(mu_0(data),axis=1)
    K1=K1.iloc[:,:-1].subtract(mu_1(data),axis=1)
    #np.sum(K0.iloc[:,:-1], axis=0)-mu_0(data)
    Sigma_0 = np.zeros((11,11))
    Sigma_1 = np.zeros((11,11))
    for i in range(len(K0)):
        Sigma_0 += np.outer(K0.iloc[i],K0.iloc[i])
        Sigma_1 += np.outer(K1.iloc[i],K1.iloc[i])
    return (1/(len(data)-2))*(Sigma_0+Sigma_1)
print(sigma(data))
'''better option for mu
#estimate means within each class
#len(K) represents the number of examples in class 0 in the training set
def mu_0(A):    
    #K consists of all vectors with label 0
    K=A[A[:,-1]==0.]
    return (1/(len(K)))*np.sum(K[:,:-1], axis=0)
def mu_1(A):
    #K consists of all vectors with label 1
    K=A[A[:,-1]==1.]
    return (1/(len(K)))*np.sum(K[:,:-1], axis=0)
print(mu_0(L1))
print(mu_1(L1))'''