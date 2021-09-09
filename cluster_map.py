import numpy as np
import pickle
#import os
import matplotlib.pyplot as plt
import seaborn as sns
#from random import randint
#from scipy import stats
#from sklearn import metrics

cancer='brca'

W=pickle.load( open( cancer+".p", "rb" ) )
feats=W['mfeats']
y=W['lab']
isffpe=np.squeeze(np.asarray(W['isffpe']))
feats=feats[isffpe<1,:]
y=y[isffpe<1]

Yl=[]
for i in range(len(y)):
    if y[i]<1:
        Yl.append('g')
    else:
        Yl.append('orange')


V=np.var(feats,axis=0)
v0=np.var(feats[y==0],axis=0)
v1=np.var(feats[y==1],axis=0)

nt=len(feats)
n0=np.sum(y==0)
n1=np.sum(y==1)

lpg=-0.5*(n0*np.log((n0-1)*v0)+n1*np.log((n1-1)*v1)-nt*np.log((nt-1)*V))
lpg[V<np.percentile(V,50)]=-10000
smones=np.argsort(lpg)
smones=np.flip(smones)

plt.figure(dpi=1200)
g1=sns.clustermap(feats[:,smones[0:100]], row_colors=Yl,vmin=0,vmax=1,cmap='Reds')
plt.savefig(cancer+'_cmap.png')