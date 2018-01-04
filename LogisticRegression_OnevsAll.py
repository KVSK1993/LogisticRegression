# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:15:11 2017

@author: Karan Vijay Singh
"""
import numpy as np

data_batch_1='C:/Users/Jasdeep/Downloads/cifar-10-python-1/cifar-10-batches-py/data_batch_1'
data_batch_2='C:/Users/Jasdeep/Downloads/cifar-10-python-1/cifar-10-batches-py/data_batch_2'
data_batch_3='C:/Users/Jasdeep/Downloads/cifar-10-python-1/cifar-10-batches-py/data_batch_3'
data_batch_4='C:/Users/Jasdeep/Downloads/cifar-10-python-1/cifar-10-batches-py/data_batch_4'
data_batch_5='C:/Users/Jasdeep/Downloads/cifar-10-python-1/cifar-10-batches-py/data_batch_5'
test_batch='C:/Users/Jasdeep/Downloads/cifar-10-python-1/cifar-10-batches-py/test_batch'

#Reading of training data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def createdata(data_batch):
    dictinput1=unpickle(data_batch)
    for i in dictinput1:
        kstr=i.decode("utf-8")
        if(kstr == 'data'):
            training_data=dictinput1[i]
        if(kstr == 'labels'):
            labelList=dictinput1[i]#np.concatenate(labelasArray
            training_result = np.asarray(labelList).reshape(len(labelList),1)                          
    return training_data,training_result

training_data,training_result=createdata(data_batch_1)

training_data_temp,training_result_temp=createdata(data_batch_2)
training_result = np.append(training_result, training_result_temp, axis=0)
training_data = np.append(training_data, training_data_temp, axis=0)

training_data_temp,training_result_temp=createdata(data_batch_3)
training_result = np.append(training_result, training_result_temp, axis=0)
training_data = np.append(training_data, training_data_temp, axis=0)

training_data_temp,training_result_temp=createdata(data_batch_4)
training_result = np.append(training_result, training_result_temp, axis=0)
training_data = np.append(training_data, training_data_temp, axis=0)

training_data_temp,training_result_temp=createdata(data_batch_5)
training_result = np.append(training_result, training_result_temp, axis=0)
training_data = np.append(training_data, training_data_temp, axis=0)

print("training_data",training_data.shape,training_result.shape)
test_data,test_result=createdata(test_batch)

       
#function to calculate h vector
def hCalculate(weight_array,training_data,training_result):
    Xw=np.dot(training_data,weight_array) #(n,d)*(d,1)=n,1
    Ywx=training_result*Xw
    return 1/(1+np.exp(Ywx))
    
datapts=training_data.shape[0]
dim=training_data.shape[1]
identityMatrix=np.identity(dim)

#function to calculate positive and negative points
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

#function to calculate gradient and hessian
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
        diffWtVector=np.subtract(oldWtVector,updatedwtVector)
        #if abs(np.dot(updatedwtVector.T,updatedwtVector)-np.dot(oldWtVector.T,oldWtVector))<tol:
        if(abs(np.linalg.norm(diffWtVector))<tol):
            break;
        #print("final wt",updatedwtVector)
    return updatedwtVector

#function to generate training data for one class as 1 and other all as -1
def Correspondingclassone(training_reslt,classNumber):
    for i in range(training_reslt.shape[0]):
        if(training_reslt[i] != classNumber):
            training_reslt[i]=-1
        else:
            training_reslt[i]=1
    return training_reslt 

NoOfClasses=10
weightVectorMatrix=np.full((training_data.shape[1],NoOfClasses),0)#10 columns and rows=no. of features i.e.3072

for i in range(0,NoOfClasses):
    training_result_Copy = np.copy(training_result)
    train_result=Correspondingclassone(training_result_Copy,i)
    #print("Total class =",(train_result==1).sum())
    finalweight=logisticRegression(training_data,train_result,labda)
    if i==0:
        weightVectorMatrix=finalweight
    else:
        weightVectorMatrix=np.concatenate((weightVectorMatrix,finalweight), axis = 1)
    
mistakeCnt=0
#making predictions
for i in range(test_data.shape[0]):
    value=np.dot(test_data[i],weightVectorMatrix)
    finalclass = np.argmax(value)#gives index of the array with max value which is the prdicted class os xi
    if(test_result[i]!=finalclass):
        mistakeCnt+=1
             
print("Number of mistakes",mistakeCnt)
print("Error rate",(mistakeCnt/test_data.shape[0])*100)  
            