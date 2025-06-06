#anomalii
import pandas as pd
df = pd.read_csv("file.csv")
legit = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

legit = legit.drop(["Class","Time"], axis = 1)
fraud = fraud.drop(["Class","Time"], axis = 1)

from sklearn.decomposition import PCA

pca = PCA(n_components = 26, random_state=0)
legit_pca = pd.DataFrame(pca.fit_transform(legit), index = legit.index)
fraud_pca = pd.DataFrame(pca.transform(fraud), index = fraud.index)

legit_restore = pd.DataFrame(pca.inverse_transform(legit_pca),index = legit.index)
fraud_restore = pd.DataFrame(pca.inverse_transform(fraud_pca),index = fraud.index)
import numpy as np
def anomaly_cals(orig,restored):
    loss = np.sum((np.array(orig) - np.array(restored))**2,axis=1)
    return pd.Series(data = loss, index = orig.index)

legit_calc = anomaly_cals(legit, legit_restore)
fraud_calc = anomaly_cals(fraud, fraud_restore)

import matplotlib.pyplot as plt
ax, = plt.subplots(1,2,share='col',share='row')

ax[0].plot(legit_calc)
ax[1].plot(fraud_calc)

th = 180
legit_TRUE = legit_calc[legit_calc<th].count()
legit_FALSE = legit_calc[legit_calc>=th].count()

fraud_TRUE = legit_calc[fraud_calc<th].count()
fraud_FALSE = legit_calc[fraud_calc>=th].count()

#reccurent NN

from fastai.text.all import *
pah = untar_data(URLs.HUMAN_NUMBERS)
linea = L()
with open('train.txt') as f:
    lines += L(*f.readlines())

text = " ".join()([l.strip() for l in lines])

tokens = text.split(" ")
vocab = L(*tokens).unique()
word2index = {w: i for i, w in enumerate(vocab)}
nums = L(word2index[i] for i in tokens)
seq = L((tokens[i : i + 3], tokens[i + 3]) for i in range(0, len(tokens) - 4, 3))
seq = L((nums[i : i + 3], nums[i + 3]) for i in range(0, len(nums) - 4, 3))
print(seq[:10])

bs = 64
cut = int(len(seq) * 0.8)
dls = DataLoaders.from_dsets(seq[:cut], seq[cut:], bs=bs, shuffle=False)

class Model1(Module):
    def __init__(self,vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden,n_hidden)
        self.h_o=nn.linear(n_hidden,vacab_sz)
    def forward(self,x):
        h = F.relu(self.h_h(self.i_h(x[:,0])))
        h = h + self.i_h(x[:,1])
        h = F.relu(self.h_h(h))
        h = h + self.i_h(x[:, 2])
        h = F.relu(self.h_h(h))
        return self.h_o(h)
learn = Learner (dls, Model1(len(vocab),bs), loss = F.cross_entropy, metrics = accuracy)

learn.fit_one_cycle(4,0.001)
n = 0
count = torch.zeros(len(vocab))

for x,y in dls.valid:
    n +=y.shape[0]
    for i in range_of(vocab):
        counts[i] += (y == 1).long().sum()

index = torch.argmax(counts)

