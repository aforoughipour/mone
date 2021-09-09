import numpy as np
#import h5py
import pickle
#from scipy.io import loadmat, savemat
from scipy import stats
import statsmodels.stats.multitest as smt
from scipy.special import loggamma as gammaln
import matplotlib.pyplot as plt
from venn import venn
cmaps = ["cool", list("rgb"), "plasma", "viridis", "Set1"]


cancer='brca'

fdim=2048
alfa=0.05


C=0.1
pg=0.5
pgi=pg/(1-pg)
mpi=np.sqrt(2*3.141592653589793)

W= pickle.load( open(cancer+".p", "rb" ) )
    
feats=W['mfeats']
y=W['lab']
sname=W['sname']
isffpe=np.asarray(W['isffpe'])
    
   
feats=feats[isffpe==0,:]
y=y[isffpe==0]

x0=np.squeeze(feats[y==0,:])
x1=np.squeeze(feats[y==1,:])


#####t-test
(T, P)=stats.ttest_ind(x0, x1, axis=0, equal_var=False)
mf_ttest=np.argsort(P)
Ql=smt.multipletests(P,alpha=alfa,method='fdr_bh')
Q=Ql[1]
MFT=np.squeeze(np.where(Q<alfa))
print('number t-test Diff mones is '+str(np.sum(Q<alfa)))

####kstest
ksP=np.zeros((fdim,))
for i in range(fdim):
    (D, P)=stats.ks_2samp(x0[:,i], x1[:,i])
    ksP[i]=P
mf_ks=np.argsort(ksP)
Ql=smt.multipletests(ksP,alpha=alfa,method='fdr_bh')
Q=Ql[1]
MFK=np.squeeze(np.where(Q<alfa))
print('number ks test Diff mones is '+str(np.sum(Q<alfa)))


##################################
##########################WRS
wrP=np.zeros((fdim,))
for i in range(fdim):
    (R, P)=stats.ranksums(x0[:,i], x1[:,i])
    wrP[i]=P
mf_wr=np.argsort(wrP)
Ql=smt.multipletests(wrP,alpha=alfa,method='fdr_bh')
Q=Ql[1]
MFW=np.squeeze(np.where(Q<alfa))
print('number WRS test Diff mones is '+str(np.sum(Q<alfa)))

####OBF
v0=np.var(x0,axis=0)
v1=np.var(x1,axis=0)
vt=np.var(np.concatenate((x0,x1),axis=0),axis=0)

nt=len(y)
n0=np.sum(y<1)
n1=np.sum(y>0)


lpg=-0.5*(n0*np.log((n0-1)*v0)+n1*np.log((n1-1)*v1)-nt*np.log((nt-1)*vt))
lqg=np.log(C*pgi*mpi)+np.log(nt/(n0*n1))+gammaln(0.5*n0)+gammaln(0.5*n1)-gammaln(0.5*nt)
lpg=lpg+lqg
hg=np.exp(lpg)
posg=hg/(1+hg)
posg[np.isnan(posg)]=1


mf=np.argsort(posg)
mf=np.flip(mf)
sP=posg[mf]


mfdr=np.cumsum(1-sP)/(1+np.arange(fdim))

Nf=0
for i in range(fdim):
    if mfdr[i]<alfa:
        Nf=i+1

print('number FDR-OBF Diff mones is '+str(Nf))

MFO=np.squeeze(mf[1:Nf])

D={}
D['t-test']=set(list(MFT))
D['KS']=set(list(MFK))
D['WRS']=set(list(MFW))
D['OBF']=set(list(MFO))

plt.figure(dpi=1200)
venn(D, cmap='rainbow', fontsize=12, legend_loc="upper left")
plt.savefig('venn_'+cancer+'_frozen_rawnum.png')

plt.figure(dpi=1200)
venn(D, cmap='rainbow', fmt="{percentage:.1f}%" , fontsize=12, legend_loc="upper left")
plt.savefig('venn_'+cancer+'_frozen_ratio.png')