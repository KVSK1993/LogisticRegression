# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:11:41 2017

@author: Karan VIjay SIngh
"""
import numpy as np
import pandas as pd
import timeit
#starting time of program
startTime = timeit.default_timer()

#Reading of training data
training_data = pd.read_csv('train_X_dog_cat.csv',header=None).as_matrix()
training_result = pd.read_csv('train_y_dog_cat.csv',header=None).as_matrix()
#print("training_result",training_result.shape)

#reading of test data
testing_data = pd.read_csv('test_X_dog_cat.csv',header=None).as_matrix()
testing_result = pd.read_csv('test_y_dog_cat.csv',header=None).as_matrix()



def hCalculate(weight_array,training_data,training_result):
    Xw=np.dot(training_data,weight_array) #(n,d)*(d,1)=n,1
    Ywx=training_result*Xw
    return 1/(1+np.exp(Ywx))
    

#print("H",Hvector)
datapts=training_data.shape[0]
dim=training_data.shape[1]
identityMatrix=np.identity(dim)

def calculateNpositiveNnegative(training_result):
    npositive=(training_result==1).sum()
    nnegative=training_result.shape[0]-npositive
    #print("+ve cnt",npositive)
    nVector=np.full((training_result.shape[0],1),0.0)
    for i in range(training_result.shape[0]):
        yValue=training_result[i]
        nVector[i]=1/(npositive*((1+yValue)/2) + nnegative*((1-yValue)/2))
    #print("nvector",nVector)
    return nVector

def CalGradientandHessian(training_data,training_result,weight_array,nVectorCoeff,labda):
    Hvector=hCalculate(weight_array,training_data,training_result) 
    #print("Hvector",Hvector.shape)
    HvectorY=(-1)*nVectorCoeff*Hvector*training_result
    gradient=np.dot(training_data.T,HvectorY)
    #print("w shape",weight_array.shape)
    gradient=np.add(gradient,2*labda*weight_array)
    #print(gradient.shape)
    oneminusHvector=1-Hvector
    temp=nVectorCoeff*oneminusHvector*Hvector
    hessian=temp*training_data#n,d
    #print("temp",temp.shape)
    hessian=np.dot(hessian.T,training_data)
    hessian=np.add(hessian,2*labda*identityMatrix)
    #print("hesian",hessian.shape)
    return gradient,hessian



tol=10**(-4)
labda=1000


def logisticRegression(training_data,training_result,labda):
    weight_array=np.full((training_data.shape[1],1),0)
    updatedwtVector=weight_array
    nVectorCoeff=calculateNpositiveNnegative(training_result)
    for i in range(50):
        #print("Iteration",i+1)
        oldWtVector=updatedwtVector    
        grad,hess=CalGradientandHessian(training_data,training_result,oldWtVector,nVectorCoeff,labda)
        weight_dash = np.linalg.solve(hess, grad)
        #print("weight_dash",weight_dash.shape)
        updatedwtVector=np.subtract(oldWtVector,weight_dash)
        #print("updatedwtVector",updatedwtVector.shape)
        #print("Weight",np.dot(updatedwtVector.T,updatedwtVector))
        diffWtVector=np.subtract(oldWtVector,updatedwtVector)
        #if abs(np.dot(updatedwtVector.T,updatedwtVector)-np.dot(oldWtVector.T,oldWtVector))<tol:
        if(abs(np.linalg.norm(diffWtVector))<tol):
            break;
        #print("final wt",updatedwtVector)
    return updatedwtVector

finalweight=logisticRegression(training_data,training_result,labda)

mistakeCnt=0
#making predictions
for i in range(testing_data.shape[0]):
    if(np.dot(testing_data[i],finalweight)>=0):
        if(testing_result[i]!=1):
            mistakeCnt+=1
    else:
        if(testing_result[i]!=-1):
             mistakeCnt+=1
             
print("Number of mistakes",mistakeCnt)
print("Error rate",(mistakeCnt/testing_data.shape[0])*100)