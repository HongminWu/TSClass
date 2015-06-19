import numpy as np
from numpy import random

def readFileLines(fileName, sLine, eLine):
	with open(fileName) as f:
		lines = f.readlines();
	ans = []
	for i in range(sLine, eLine):
		ans.append(lines[i].split())
	for i in range(len(ans)):
		for j in range(len(ans[i])):
			ans[i][j] = float(ans[i][j])
	return ans

labels = []
with open("MotionData2/labels.txt") as f:
	lines = f.readlines()
for line in lines:
	labels.append(line.split())

for i in range(len(labels)):
	for j in range(len(labels[i])):
		labels[i][j] = int(labels[i][j])
	labels[i][2] = labels[i][2] - 1

#Generating Training Sample
xTrain = []
yTrain = []
lenLabel = len(labels)
for ACT in range(12):
	for TIME in range(500):
		print ACT, TIME
		l = random.randint(lenLabel)
		while labels[l][2]!=ACT or labels[l][4]-labels[l][3]<138:
			l = random.randint(lenLabel)
		if labels[l][0]<10: 
			str0 = "0"+str(labels[l][0])
		else:
			str0 = str(labels[l][0])
		if labels[l][1]<10: 
			str1 = "0"+str(labels[l][1])
		else:
			str1 = str(labels[l][1])


		accFile = "MotionData2/acc_exp" + str0 + "_user" + str1 + ".txt"
		gyroFile = "MotionData2/gyro_exp" + str0 + "_user" + str1 + ".txt"
		startLine = random.randint(labels[l][3]+2, labels[l][4]-130)
		accData = readFileLines(accFile, startLine, startLine+128)
		gyroData = readFileLines(gyroFile, startLine, startLine+128)
		x = []
		for i in range(128):
			x.append([accData[i][0], accData[i][1], accData[i][2], gyroData[i][0], gyroData[i][1], gyroData[i][2]])
		xTrain.append(list(x))
		yTrain.append(labels[l][2])

# np.save("MotionData2/xTrain.npy", np.array(xTrain))
# np.save("MotionData2/yTrain.npy", np.array(yTrain))

np.save("MotionData2/xTest.npy", np.array(xTrain))
np.save("MotionData2/yTest.npy", np.array(yTrain))

