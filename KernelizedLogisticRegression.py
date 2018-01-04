# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:15:11 2017

@author: Karan Vijay Singh
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing 

labda=1000
noOfIterations=50
sigma =2000

print("Lambda=",labda)
print("sigma=",sigma)
#Reading of training data
training_data = pd.read_csv('train_X_dog_cat.csv',header=None).as_matrix()
training_result = pd.read_csv('train_y_dog_cat.csv',header=None).as_matrix()
print("training_result",training_result.shape)

training_data=preprocessing.normalize(training_data)
dataPts=training_data.shape[0]
dims=training_data.shape[1]
#reading of test data
testing_data = pd.read_csv('test_X_dog_cat.csv',header=None).as_matrix()
testing_result = pd.read_csv('test_y_dog_cat.csv',header=None).as_matrix()
testing_data=preprocessing.normalize(testing_data)

reshapedtesting_result = np.reshape(testing_result,(1,testing_result.shape[0]))
#print("reshapedtesting_result shape", reshapedtesting_result.shape)

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


#square matrix to be used for calculation and also gram matrix for linear kernel
XXT = np.dot(training_data,training_data.T)
print("XXT shape",XXT.shape) #1953*1953

diagonalVector=np.diagonal(XXT)#extracting diagonal elements because they are l2 norms
reshapeDiagonalVector=diagonalVector.reshape(1,diagonalVector.shape[0])

#evaluating entire test data at once for linear kernel
def TestingForLinearKernel(sumTrainingVectors,testing_data):
    return(np.dot(sumTrainingVectors,testing_data.T))
    
def gramMatrixforLinearKernel(XXT):
    return XXT

def gramMatrixforPolynomialKernel(XXT):
    return (1+XXT)**5

def gramMatrixforGaussianKernel(XXT, sigma,reshapeDiagonalVector):
    dimofMatrix=reshapeDiagonalVector.shape[1]
    tileMatrix=np.tile(reshapeDiagonalVector,(dimofMatrix,1))
    trMatrix=tileMatrix.T
    trMatrix=reshapeDiagonalVector+trMatrix
    finalGramMatrix=(-1.0/sigma)*np.subtract(trMatrix,2*XXT)
    return np.exp(finalGramMatrix)
        
def betaCalculate(gramMatrix,alpha_vector,training_result):
    gAlpha=np.dot(gramMatrix,alpha_vector)
    return 1/(1+np.exp(training_result*gAlpha))

def CalGradientandHessian(gramMatrix,nVectorCoeff,training_result,labda,alpha_vector):
    noOfDatapts=gramMatrix.shape[0]
    beta_Vector = betaCalculate(gramMatrix, alpha_vector, training_result)
    
    BetaCoeff = nVectorCoeff*beta_Vector
    gradient = (-1.0)*BetaCoeff*training_result
    gradient = np.dot(gramMatrix,gradient)
    interValue = (2.0)*labda*(np.dot(gramMatrix,alpha_vector))
    gradient = np.add(gradient,interValue)
    #Hessian
    identityMatrix = np.identity(noOfDatapts)
    changed_Beta_Vector = 1-beta_Vector
    hessian = BetaCoeff*changed_Beta_Vector*gramMatrix
    hessian = np.add(np.transpose(hessian), 2*labda*identityMatrix)
    hessian = np.dot(hessian,gramMatrix)
    return (gradient,hessian)

def logisticRegression(training_data, training_result,labda, gramMatrix):
    alpha_vector = np.full((dataPts,1),0.0)
    finalAlpha = alpha_vector
    
    nVectorCoeff=calculateNpositiveNnegative(training_result)
    for k in range(noOfIterations):
        oldAlpha = finalAlpha
        (gradient,hessian) = CalGradientandHessian(gramMatrix,nVectorCoeff,training_result,labda, alpha_vector)
        alphaDash = np.linalg.solve(hessian,gradient)
        finalAlpha = np.subtract(oldAlpha,alphaDash)
        diffAlphaVectors = np.subtract(oldAlpha,finalAlpha)
        if(abs(np.linalg.norm(diffAlphaVectors)) < 10**-4):
            break;
            
    return finalAlpha

linearAlpha = logisticRegression(training_data, training_result,labda,gramMatrixforLinearKernel(XXT))
print("Alpha for Linear Kernel", linearAlpha.shape, np.dot(linearAlpha.T,linearAlpha))
polynomialAlpha = logisticRegression(training_data, training_result,labda,gramMatrixforPolynomialKernel(XXT))
print("Alpha for Polynomial Kernel", polynomialAlpha.shape, np.dot(polynomialAlpha.T,polynomialAlpha))
gaussianAlpha = logisticRegression(training_data, training_result,labda,gramMatrixforGaussianKernel(XXT,sigma,reshapeDiagonalVector))
print("Alpha for Gaussian Kernel", gaussianAlpha.shape, np.dot(gaussianAlpha.T,gaussianAlpha))

#For testing will Reuse it again in different kernels 
XXtestTr = np.dot(training_data,np.transpose(testing_data))
dim = testing_result.shape[0]

#For linear
linearKPreds = np.dot(np.transpose(linearAlpha),XXtestTr) # 1,no of test points vector
linearKPreds[linearKPreds>=0] = 1
linearKPreds[linearKPreds<0] = -1

linearMisclassification = ((np.subtract(linearKPreds,reshapedtesting_result))!=0).sum()

print("Misclassification in Linear Kernel", linearMisclassification, linearMisclassification/dim)

# For polynoial
polyKPred = np.dot(np.transpose(polynomialAlpha),(1+XXtestTr)**5) # 1,no of test points
polyKPred[polyKPred>=0] = 1
polyKPred[polyKPred<0] = -1

polyMisclassification = ((np.subtract(polyKPred,reshapedtesting_result))!=0).sum()
print("Misclassification in Polynomial Kernel", polyMisclassification, polyMisclassification/dim)

#For Gaussian
tileMatrix = np.tile(reshapeDiagonalVector,(dim,1))
trMatrix = tileMatrix.T## n*t matrix
l2TestData = (np.square(testing_data).sum(axis=1)).reshape(1,dim)
trMatrix = l2TestData + trMatrix # broadcasting
sub = np.subtract(trMatrix,2*XXtestTr)
testGMatrix = (-1.0/sigma)*(sub)
testGMatrix = np.exp(testGMatrix)  # n,t 
gaussKPreds = np.dot(np.transpose(gaussianAlpha),testGMatrix)

gaussKPreds[gaussKPreds>=0] = 1
gaussKPreds[gaussKPreds<0] = -1
gaussMisclassification = ((np.subtract(gaussKPreds,reshapedtesting_result))!=0).sum()

print("Misclassification for Gaussian Kernel", gaussMisclassification, gaussMisclassification/dim)

