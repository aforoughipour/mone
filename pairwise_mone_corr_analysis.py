import numpy as np
import pickle
from scipy import stats
import statsmodels.stats.multitest as smt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf

cancer='brca'
include_ffpe=1  ##0: only forzen slides, 1: include both FFPE and frozen slide are included in the analysis as well


fdim=2048
T=10**-3
T2=10**-10

    
W= pickle.load( open(cancer+".p", "rb" ) )
        
feats=W['mfeats']
y=W['lab']
isffpe=np.asarray(W['isffpe'])
        
if include_ffpe==0:
    feats=feats[isffpe==0,:]
    y=y[isffpe==0]

x0t=feats[y<1,:]
x1t=feats[y>0,:]
        
n0=len(x0t)
n1=len(x1t)

v0i=1/(n0-3)
v1i=1/(n1-3)
    
c0a=LedoitWolf().fit(x0t)
c1a=LedoitWolf().fit(x1t)

cc0=c0a.covariance_
cc1=c1a.covariance_

v0=np.squeeze(np.diagonal(cc0))
v1=np.squeeze(np.diagonal(cc1))

v0=v0[:,np.newaxis]
v1=v1[:,np.newaxis]

cc0=cc0/np.sqrt(v0*v0.T)
cc1=cc1/np.sqrt(v1*v1.T)
            

    
#####################
####plot tumor cluster map

plt.figure(dpi=1200)
gt=sns.clustermap(cc1,cmap="seismic",vmin=-1, vmax=1)
plt.savefig(cancer+"_tumorclustergram.png")
plt.close('all')
    
#######################
####plot difference in correlations
cd=cc1-cc0
plt.figure(dpi=1200)
gd=sns.clustermap(cd,cmap="seismic",vmin=-2, vmax=2)
plt.savefig(cancer+"_differentialclustergram.png")
plt.close('all')
    
# #######hypothesis tests how many significant??????
Fcc1=0.5*np.log((1+cc1+T)/(1-cc1+T))
Fcc0=0.5*np.log((1+cc0+T)/(1-cc0+T))
Fcd=Fcc1-Fcc0
    
combs=0.5*fdim*(fdim-1)
combs=int(combs)
cmat1=np.zeros((3,combs))
cmatd=np.zeros((3,combs))
    
cnt=0
for i in range(2048):
    for j in range(i+1,2048):
            
        cmat1[0,cnt]=Fcc1[i,j]
        cmat1[1,cnt]=i
        cmat1[2,cnt]=j
        
        cmatd[0,cnt]=Fcd[i,j]
        cmatd[1,cnt]=i
        cmatd[2,cnt]=j
        
        
        cnt=cnt+1
                
       
Zsc=np.squeeze(cmat1[0,:])/(np.sqrt(v1i))   
Pv=2*stats.norm.sf(np.abs(Zsc))
Pv[np.isnan(Pv)]=T2
Pv[Pv<T2]=T2
    
Q=smt.multipletests(Pv,alpha=0.05, method='fdr_bh')
Q=np.squeeze(Q[1])
Nsig=np.sum(Q<0.05)

print('number of sig. mone corrs across tumors in '+cancer+' is '+str(Nsig))
print('percent of sig. mone corrs across tumors in '+cancer+' is '+str(100*Nsig/combs))

Zsc=np.squeeze(cmatd[0,:])/(np.sqrt(v0i+v1i))   
Pv=2*stats.norm.sf(np.abs(Zsc))
Pv[np.isnan(Pv)]=T2
Pv[Pv<T2]=T2
    
Q=smt.multipletests(Pv,alpha=0.05, method='fdr_bh')
Q=np.squeeze(Q[1])
Nsig=np.sum(Q<0.05)

print('number of sig. diff mone corrs across slides in '+cancer+' is '+str(Nsig))
print('percent of sig. mone corrs across slides in '+cancer+' is '+str(100*Nsig/combs))
