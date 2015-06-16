import numpy as np

y = np.load("MotionData/Ytrain.npy")
yF = np.load("MotionData/yTrainF.npy")
yC = np.load("MotionData/yTrainC.npy")
yL = np.load("MotionData/xTrainL.npy")
lF = 0
lC = 0

for i in range(len(y)):
	if not yL[i]:
		if y[i] != yC[lC]:
			print "FUCK!"
			break
		lC = lC + 1
	else:
		if y[i] != yF[lF]:
			print "FUCK!"
			break
		lF = lF + 1
else:
	print "NICE!"

print "I AM A STUPID LINE"
