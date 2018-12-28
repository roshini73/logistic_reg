import numpy as np
import csv
import operator
import math

def main():
  info, x, y = loadCsvData('ancestry-train.txt')
  testInfo, testX, testY = loadCsvData('ancestry-test.txt')
  thetas = trainLR(info, x, y, 10000, 0.0001)
  guessY = testData(testInfo, testX, testY, thetas)
  printAcc(testY, guessY)

#prints accuracy of classification
def printAcc(y, guessY):
  count = [0, 0]
  corr = [0, 0]
  for i in range(len(y)):
    count[y[i]] += 1
    if y[i] == 1 and guessY[i] == 1:
      corr[1] += 1
    elif y[i] == 0 and guessY[i] == 0:
      corr[0] += 1
  print "Class 0: correctly classified", corr[0], "out of", count[0], "tested"
  print "Class 1: correctly classified", corr[1], "out of", count[1], "tested"
  print "Overall: correctly classified", (corr[0] + corr[1]), "out of", (count[0] + count[1]), "tested"
  print "Accuracy =", (float(corr[0] + corr[1])/(count[0] + count[1]))


#predicts values for data in test file given array with input data from test file
def testData(info, x, y, thetas):
  guessY = np.array([])
  guessY = guessY.astype(int)
  for i in range (len(y)):
    wSum = 0
    xs = np.array([1])
    rest = np.array(x[i])
    xs = np.concatenate((xs, rest), axis = None)
    wSum += (np.sum(np.multiply(xs, thetas)))
    prob1 = 1/ (1+math.exp(-wSum))
    prob0 = 1 - prob1
    if (prob1 > prob0):
      guessY = np.append(guessY, 1)
    elif (prob0 > prob1):
      guessY = np.append(guessY, 0)
    else :
      r = np.random.randint(2)
      if r == 0:
        guessY = np.append(guessY, 0)
      else:
        guessY = np.append(guessY, 1)
  return guessY


#calculates parameter weights
def trainLR(info, x, y, maxIter, eta):
  numOfInput = info[0]
  thetas = np.zeros((numOfInput+1))
  for i in range(maxIter):
    grads = np.zeros((numOfInput+1))
    for j in range(len(y)):
      xs = np.array([1])
      rest = np.array(x[j,:])
      xs = np.concatenate((xs, rest), axis = None)
      thetaTx = np.dot(thetas, xs)
      thetaTx = (math.exp(-thetaTx))
      thetaTx = 1/(1+thetaTx)
      grads += xs * (y[j] - thetaTx)
    thetas += eta * grads
  return thetas

#loads data from provided text file into array
def loadCsvData(fileName):
  input = np.array([])
  output = np.array([])
  info = np.array([])
  with open(fileName) as f:
    reader = csv.reader(f)
    for row in reader:
      row = row[0].split(':')
      if len(row) == 1:
        info = np.append(info, row[0])
      if len(row) > 1 :
        output = np.append(output, row[1])
        input= np.append(input, row[0].split(), axis = 0)
    input = np.reshape(input, (-1,int(info[0])))
    input = input.astype(int)
    output= output.astype(int)
    info = info.astype(int)
  return info, input, output

def printData(matrix):
  for row in matrix:
    print row


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
	main()
