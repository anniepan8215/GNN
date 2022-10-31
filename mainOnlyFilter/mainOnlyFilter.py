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
import matplotlib.pyplot as plt

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
yTrain = torch.tensor(yTrain)

xTest = torch.tensor(xTest)
yTest = torch.tensor(yTest)


################################
######## LOSS FUNCTION #########
################################

loss = nn.MSELoss()


################################
######## ARCHITECTURES #########
################################

# Graph filter

graphFilter = myModules.GraphFilter(S, 8)


################################
########### TRAINING ###########
################################

validationInterval = 5

nEpochs = 30
batchSize = 200
learningRate = 0.05

nValid = int(np.floor(0.01*nTrain))
xValid = xTrain[0:nValid,:]
yValid = yTrain[0:nValid,:]
xTrain = xTrain[nValid:,:]
yTrain = yTrain[nValid:,:]
nTrain = xTrain.shape[0]

# Declaring the optimizers for each architectures
optimizer = optim.Adam(graphFilter.parameters(), lr=learningRate)

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
lossValid = dict()
# ...and test variables
lossTestBest = dict()
lossTestLast = dict()

bestModel = dict()

lossTrain = []
lossValid = []
    
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
        xTrainBatch = xTrain[thisBatchIndices,:]
        yTrainBatch = yTrain[thisBatchIndices,:]

        if (epoch * nBatches + batch) % validationInterval == 0:
            print("")
            print("    (E: %2d, B: %3d)" % (epoch+1, batch+1),end = ' ')
            print("")
        
       
        # Reset gradients
        graphFilter.zero_grad()

        # Obtain the output of the architectures
        yHatTrainBatch = graphFilter(xTrainBatch)

        # Compute loss
        lossValueTrain = loss(yHatTrainBatch, yTrainBatch)

        # Compute gradients
        lossValueTrain.backward()

        # Optimize
        optimizer.step()
        
        lossTrain += [lossValueTrain.item()]
        
        # Print:
        if (epoch * nBatches + batch) % validationInterval == 0:
            with torch.no_grad():
                # Obtain the output of the GNN
                yHatValid = graphFilter(xValid)
    
            # Compute loss
            lossValueValid = loss(yHatValid, yValid)
            
            lossValid += [lossValueValid.item()]

            print("\t Graph Filter: %6.4f [T]" % (
                    lossValueTrain) + " %6.4f [V]" % (
                    lossValueValid))
            
            # Saving the best model so far
            if len(lossValid) > 1:
                if lossValueValid <= min(lossValid):
                    bestModel =  copy.deepcopy(graphFilter)
            else:
                bestModel =  copy.deepcopy(graphFilter)
                    
        batch+=1
        
    epoch+=1
    
print("")

################################
############# PLOT #############
################################
 
plt.plot(lossTrain)
plt.ylabel('Training loss')
plt.xlabel('Step')
plt.show()
   
################################
########## EVALUATION ##########
################################

print("Final evaluation results")

with torch.no_grad():
    yHatTest = graphFilter(xTest)
lossTestLast = loss(yHatTest, yTest)
lossTestLast = lossTestLast.item()
with torch.no_grad():
    yHatTest = bestModel(xTest)
lossTestBest = loss(yHatTest, yTest)
lossTestBest = lossTestBest.item()

print(" Graph Filter: %6.4f [Best]" % (
                    lossTestBest) + " %6.4f [Last]" % (
                    lossTestLast))
 