# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:51:49 2020

@author: Luana Ruiz
"""

"""
LAB 2: SOURCE LOCALIZATION
"""

#\\\ Standard libraries:
import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib
matplotlib.use('TkAgg')

#\\\ Own libraries:
import data as data
import myModules as myModules


################################
####### DATA GENERATION ########
################################

N = 50 # number of nodes

S = data.sbm(n=N)

S = data.normalize_gso(S)

nTrain = 2000
nTest = 100

z = data.generate_diffusion(gso=S, n_samples=nTrain+nTest)

x, y = data.data_from_diffusion(z)

trainData, testData = data.split_data(x, y, (nTrain,nTest))
xTrain = trainData[0]
yTrain = trainData[1]
xTest = testData[0]
yTest = testData[1]

xTrain = torch.tensor(xTrain)
xTrain = xTrain.reshape([-1,1,N])
yTrain = torch.tensor(yTrain)
yTrain = yTrain.reshape([-1,1,N])

xTest = torch.tensor(xTest)
xTest = xTest.reshape([-1,1,N])
yTest = torch.tensor(yTest)
yTest = yTest.reshape([-1,1,N])


################################
######## LOSS FUNCTION #########
################################

loss = nn.MSELoss()


################################
######## ARCHITECTURES #########
################################

architectures = dict()

# Linear parametrization    
linearParam = torch.nn.Linear(N,N, bias = True)
architectures['LinearLayer  '] = linearParam

# Fully connected neural network   
fcNet = nn.Sequential(torch.nn.Linear(N,25, bias=True), nn.ReLU(), torch.nn.Linear(25,N, bias=True), nn.ReLU())
architectures['FCLayer      '] = fcNet

# Multi-feature graph filter
MLgraphFilter = nn.Sequential(myModules.GraphFilter(S,8,1,32,True), myModules.GraphFilter(S,1,32,1,True))
architectures['MFGraphFilter'] = MLgraphFilter

# GNN, 1 layer
GNN1Ly = myModules.GNN(S,2,[8,1],[1,32,1],nn.ReLU(),True)
architectures['GNN 1 layer  '] = GNN1Ly

# GNN, 2 layers
GNN2Ly = myModules.GNN(S,3,[5,5,1],[1,16,4,1],nn.ReLU(),True)
architectures['GNN 2 layer  '] = GNN2Ly


################################
########### TRAINING ###########
################################

validationInterval = 5

nEpochs = 30
batchSize = 200
learningRate = 0.05

nValid = int(np.floor(0.01*nTrain))
xValid = xTrain[0:nValid,:,:]
yValid = yTrain[0:nValid,:,:]
xTrain = xTrain[nValid:,:,:]
yTrain = yTrain[nValid:,:,:]
nTrain = xTrain.shape[0]

# Declaring the optimizers for each architectures
optimizers = dict()
for key in architectures.keys():
    optimizers[key] = optim.Adam(architectures[key].parameters(), lr=learningRate)

if nTrain < batchSize:
    nBatches = 1
    batchSize = [nTrain]
elif nTrain % batchSize != 0:
    nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
    batchSize = [batchSize] * nBatches
    while sum(batchSize) != nTrain:
        batchSize[-1] -= 1
else:
    nBatches = np.int(nTrain/batchSize)
    batchSize = [batchSize] * nBatches
batchIndex = np.cumsum(batchSize).tolist()
batchIndex = [0] + batchIndex

epoch = 0 # epoch counter

# Store the training...
lossTrain = dict()
costTrain = dict()
lossValid = dict()
costValid = dict()
# ...and test variables
lossTestBest = dict()
costTestBest = dict()
lossTestLast = dict()
costTestLast = dict()

bestModel = dict()

for key in architectures.keys():
    lossTrain[key] = []
    costTrain[key] = []
    lossValid[key] = []
    costValid[key] = []
    
while epoch < nEpochs:
    randomPermutation = np.random.permutation(nTrain)
    idxEpoch = [int(i) for i in randomPermutation]
    print("")
    print("Epoch %d" % (epoch+1))

    batch = 0 
    
    while batch < nBatches:
        # Determine batch indices
        thisBatchIndices = idxEpoch[batchIndex[batch]
                                    : batchIndex[batch+1]]
        
        # Get the samples in this batch
        xTrainBatch = xTrain[thisBatchIndices,:,:]
        yTrainBatch = yTrain[thisBatchIndices,:,:]

        if (epoch * nBatches + batch) % validationInterval == 0:
            print("")
            print("    (E: %2d, B: %3d)" % (epoch+1, batch+1),end = ' ')
            print("")
        
        for key in architectures.keys():
            # Reset gradients
            architectures[key].zero_grad()

            # Obtain the output of the architectures
            yHatTrainBatch = architectures[key](xTrainBatch)

            # Compute loss
            lossValueTrain = loss(yHatTrainBatch.squeeze(), yTrainBatch.squeeze())
    
            # Compute gradients
            lossValueTrain.backward()
    
            # Optimize
            optimizers[key].step()

            costValueTrain = lossValueTrain.item()
            
            lossTrain[key] += [lossValueTrain.item()]
            costTrain[key] += [costValueTrain]
            
            # Print:
            if (epoch * nBatches + batch) % validationInterval == 0:
                with torch.no_grad():
                    # Obtain the output of the GNN
                    yHatValid = architectures[key](xValid)
        
                # Compute loss
                lossValueValid = loss(yHatValid.squeeze(), yValid.squeeze())

                # Compute accuracy:
                costValueValid = lossValueValid.item()
                
                lossValid[key] += [lossValueValid.item()]
                costValid[key] += [costValueValid]

                print("\t" + key + ": %6.4f [T]" % (
                        costValueTrain) + " %6.4f [V]" % (
                        costValueValid))
                
                # Saving the best model so far
                if len(costValid[key]) > 1:
                    if costValueValid <= min(costValid[key]):
                        bestModel[key] =  copy.deepcopy(architectures[key])
                else:
                    bestModel[key] =  copy.deepcopy(architectures[key])
                    
        batch+=1
        
    epoch+=1
    
print("")
 
   
################################
########## EVALUATION ##########
################################

print("Final evaluation results")

for key in architectures.keys():
    with torch.no_grad():
        yHatTest = architectures[key](xTest)
    lossTestLast[key] = loss(yHatTest.squeeze(), yTest.squeeze())
    costTestLast[key] = lossTestLast[key].item()
    with torch.no_grad():
        yHatTest = bestModel[key](xTest)
    lossTestBest[key] = loss(yHatTest.squeeze(), yTest.squeeze())
    costTestBest[key] = lossTestBest[key].item()
    
    print(" " + key + ": %6.4f [Best]" % (
                        costTestBest[key]) + " %6.4f [Last]" % (
                        costTestLast[key]))
 
    
################################
###### TRANSFERABILITY 1 #######
################################

N2 = 500 # number of nodes

S2 = data.sbm(n=N2)

S2 = data.normalize_gso(S2)

z2 = data.generate_diffusion(gso=S2, n_samples=nTest, n_sources=100)

x2, y2 = data.data_from_diffusion(z2)

xTest2 = x2
yTest2 = y2

xTest2 = torch.tensor(xTest2)
xTest2 = xTest2.reshape([-1,1,N2])
yTest2 = torch.tensor(yTest2)
yTest2 = yTest2.reshape([-1,1,N2])

lossTest2 = dict()
costTest2 = dict()

bestModel['GNN 1 layer  '].changeGSO(S2)
bestModel['GNN 2 layer  '].changeGSO(S2)

print("")
print("Transferability to N=500")
print("")

with torch.no_grad():
    yHatTest2 = bestModel['GNN 1 layer  '](xTest2)
lossTest2['GNN 1 layer  '] = loss(yHatTest2, yTest2)
costTest2['GNN 1 layer  '] = lossTest2['GNN 1 layer  '].item()

with torch.no_grad():
    yHatTest2 = bestModel['GNN 2 layer  '](xTest2)
lossTest2['GNN 2 layer  '] = loss(yHatTest2, yTest2)
costTest2['GNN 2 layer  '] = lossTest2['GNN 2 layer  '].item()

print(" " + "GNN 1 layer: %6.4f" % (costTest2['GNN 1 layer  ']))
print(" " + "GNN 2 layer: %6.4f" % (costTest2['GNN 2 layer  ']))


################################
###### TRANSFERABILITY 2 #######
################################

N3 = 1000 # number of nodes

S3 = data.sbm(n=N3)

S3 = data.normalize_gso(S3)

z3 = data.generate_diffusion(gso=S3, n_samples=nTest, n_sources=200)

x3, y3 = data.data_from_diffusion(z3)

xTest3 = x3
yTest3 = y3

xTest3 = torch.tensor(xTest3)
xTest3 = xTest3.reshape([-1,1,N3])
yTest3 = torch.tensor(yTest3)
yTest3 = yTest3.reshape([-1,1,N3])

lossTest3 = dict()
costTest3 = dict()

bestModel['GNN 1 layer  '].changeGSO(S3)
bestModel['GNN 2 layer  '].changeGSO(S3)

print("")
print("Transferability to N=1000")
print("")

with torch.no_grad():
    yHatTest3 = bestModel['GNN 1 layer  '](xTest3)
lossTest3['GNN 1 layer  '] = loss(yHatTest3, yTest3)
costTest3['GNN 1 layer  '] = lossTest3['GNN 1 layer  '].item()

with torch.no_grad():
    yHatTest3 = bestModel['GNN 2 layer  '](xTest3)
lossTest3['GNN 2 layer  '] = loss(yHatTest3, yTest3)
costTest3['GNN 2 layer  '] = lossTest3['GNN 2 layer  '].item()

print(" " + "GNN 1 layer: %6.4f" % (costTest3['GNN 1 layer  ']))
print(" " + "GNN 2 layer: %6.4f" % (costTest3['GNN 2 layer  ']))

