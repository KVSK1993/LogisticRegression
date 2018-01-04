# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:15:11 2017

@author: Karan Vijay Singh
"""
import numpy as np
from itertools import combinations

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
    

#print("H",Hvector)
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
        oldWtVector=updatedwtVector    
        grad,hess=CalGradientandHessian(training_data,training_result,oldWtVector,nVectorCoeff,labda)
        weight_dash = np.linalg.solve(hess, grad)
        #print("weight_dash",weight_dash.shape)
        updatedwtVector=np.subtract(oldWtVector,weight_dash)
        diffWtVector=np.subtract(oldWtVector,updatedwtVector)
        #if abs(np.dot(updatedwtVector.T,updatedwtVector)-np.dot(oldWtVector.T,oldWtVector))<tol:
        if(abs(np.linalg.norm(diffWtVector))<tol):
            break;
    return updatedwtVector

#function to generate training data for one class as 1 and other all as -1
def Correspondingclassone(training_data,training_reslt,classNumber1,classNumber2):
    inputlistTrain = list() 
    outputListTrain = list()
    for i in range(training_reslt.shape[0]):
        if(training_reslt[i] == classNumber1):
            inputlistTrain.append(training_data[i])
            outputListTrain.append(1)
        elif(training_reslt[i] == classNumber2):
             inputlistTrain.append(training_data[i])
             outputListTrain.append(-1)
    trainInputArray = np.asarray(inputlistTrain)
    trainOutputArray = np.asarray(outputListTrain).reshape(len(outputListTrain),1)
    return trainInputArray, trainOutputArray

NoOfClasses=45
weightVectorMatrix=np.full((training_data.shape[1],NoOfClasses),0)#10 columns and rows=no. of features i.e.3072

cnt=0
comb = list(combinations(range(0,10),2))

#creating 45 classifiers 
for k in range(len(comb)):
    classCb = comb[k]
    positiveC = classCb[0]
    negativeC = classCb[1]
    cnt+=1
    train_data,train_result=Correspondingclassone(training_data,training_result,positiveC,negativeC)
    finalweight=logisticRegression(train_data,train_result,labda)
    if cnt==1:
        weightVectorMatrix=finalweight
    else:
        weightVectorMatrix=np.concatenate((weightVectorMatrix,finalweight), axis = 1)
    print("weightvector shape",weightVectorMatrix.shape)


mistakeCnt=0
dims=test_data.shape[1]
#making predictions FOR TEST DATA with each of the 45 classifier
for p in range(test_data.shape[0]):
    testInput=test_data[p].reshape(dims,1)#d,1
    finaltest=np.dot(testInput.T,weightVectorMatrix)#1,45
    finaltest[finaltest>0] = 1
    finaltest[finaltest<0] = -1
    rangeI = 9
    startVal = 0
    classify_array = None
    #For Creating test matrix of 45 classifiers
    for k in range(9):
        b = finaltest[:,startVal:startVal+rangeI]
        sub = b.reshape(1,rangeI)
        if k != 0:
            appendZero = (np.zeros(k)).reshape(1,k)
            sub = np.concatenate((appendZero,sub),1)
            classify_array = np.concatenate((classify_array,sub),0)
        else:
            classify_array = sub
        startVal = startVal+rangeI    
        rangeI-=1
   
    votearray=np.sum(classify_array,axis=1)
    #print("votearray b4",votearray.shape)
    votearray=votearray.reshape(9,1)#change to 9 i.e number of classes -1
    #print("votearray after",votearray)
    for s in range(9):
        if s != 0:
            votearray[s]-=np.sum(classify_array[:,s-1])
    
    x=-np.sum(classify_array[:,8])#lastclass -1
    y = (np.array(x)).reshape(1,1)
    votearray=np.concatenate((votearray,y), axis = 0) 
    finalpredictedclass=np.argmax(votearray)
    #print("finalpredictedclass and actual class",finalpredictedclass,test_result[p])
    if(test_result[p]!=finalpredictedclass):
        mistakeCnt+=1

             
print("Number of mistakes",mistakeCnt)
print("Error rate",(mistakeCnt/test_data.shape[0])*100) 