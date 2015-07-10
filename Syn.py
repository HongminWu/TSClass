import numpy as np
import random 

templates = np.load("Synthetic2/xSyn2.npy")
templateLabels = np.load("Synthetic2/ySyn2.npy")

# templates = np.load("synthetic/dba_templates.npy")
# templateLabels = np.load("synthetic/ycluster10.npy")

nTemplate = len(templates)

print nTemplate

syn = []
synLabel = []
for ACT in range(6):
	for TIME in range(500):
		k = random.randint(0, nTemplate-1)
		while templateLabels[k]!= ACT:
			k = random.randint(0, nTemplate-1)
		synLabel.append(templateLabels[k])
		l = len(templates[k])
		pool = range(random.randint(0, 10), l)*10
		seq = sorted(random.sample(pool, l))
		newTemp = [templates[k][i] for i in seq]
		for i in range(len(newTemp)):
			for j in range(len(newTemp[i])):
				newTemp[i][j] *= random.gauss(1, 0.1)
		syn.append(newTemp)

syn = np.array(syn)
synLabel = np.array(synLabel)

# np.save("synthetic/xTrainSyn.npy", syn)
# np.save("synthetic/yTrainSyn.npy", synLabel)

# np.save("synthetic/xTestSyn.npy", syn)
# np.save("synthetic/yTestSyn.npy", synLabel)

np.save("Synthetic2/xTrainSyn2.npy", syn)
np.save("Synthetic2/yTrainSyn2.npy", synLabel)

# np.save("Synthetic2/xTestSyn2.npy", syn)
# np.save("Synthetic2/yTestSyn2.npy", synLabel)