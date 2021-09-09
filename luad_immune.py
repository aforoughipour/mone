import numpy as np
import pickle
from scipy import stats
import statsmodels.stats.multitest as smt
from sklearn.covariance import LedoitWolf
import os
import pandas as pd
#from sklearn.decomposition import PCA
#import shutil
import matplotlib.pyplot as plt
import seaborn as sns


####parameters of the model
fdim=2048
gdim=60483
Texpr=0.5  ####minimum expression of a mone or gene
T=10**-3   ####regularization for fisher transform, for numerical stabiulity
T2=10**-100 ####min threhsold for p-values, for numerical stability and nan pvalues
Tc=0.05  ###the threshold correlation for the null, i.e., null= |corr coeff| \leq Tc
dS=4000

cancers=['luad']
igenes=['CD2','CD3E','GZMA','PRF1','NKG7','GZMH','GZMK','CD247']

########################################################################
########################################################################
##data loading, commented
###instead mone and expression matrices are saved in a pickle file

# masirm='/Users/foroua/Documents/python/ML2/preproc/medsFFPE_NPT/'
# masird='/Users/foroua/Documents/python/ML/CME/SP/'
# masirc='/Users/foroua/Documents/python/ML2/immune/'
# masirp='/Users/foroua/Documents/python/ML2/'

# egmap=pd.read_csv(masirc+'eglist.csv')
# glist=list(egmap.loc[2])
# glist=glist[1:]

# X=np.zeros((dS,fdim))
# G=np.zeros((dS,gdim))
# rsname=[]

# cnt=0
# for cancer in cancers:

#     cmasird=masird+cancer+'/'
#     W=pickle.load( open( masirm+cancer+".p", "rb" ) )
    
#     cfeats=W['mfeats']
#     cy=W['lab']
#     csname=W['sname']
    
#     csnum=len(csname)
    
        
#     Gfiles=os.listdir(cmasird)


#     for i in range(csnum):
#         if (cy[i]==1) and (csname[i]+'.p' in Gfiles):
            
#             rsname.append(csname[i])
            
#             X[cnt,:]=cfeats[i,:]
#             G[cnt,:]=pickle.load( open( cmasird+csname[i]+".p", "rb" ) )
#             cnt=cnt+1

# X=X[:cnt,:]
# G=G[:cnt,:]

# G=np.log(1+G)

# snum=len(X)

# Xto_include=1*(np.sum(X>0.01,axis=0)/snum)>Texpr
# Gto1=1*(np.sum(G>0.01,axis=0)/snum>Texpr)
# Gto2=1*(np.std(G,axis=0)>0.25)
# Gto_include=1*(Gto1+Gto2==2)

# moneid=np.squeeze(np.asarray(np.where(Xto_include>0)))
# geneid=np.squeeze(np.asarray(np.where(Gto_include>0)))

# cglist=[]
# for i in range(gdim):
#     if Gto_include[i]>0:
#         cglist.append(glist[i])

# Xr=X[:,moneid]
# Gr=G[:,geneid]

# del G, egmap

# #############################################
# Tgeneinds=[]
# fin_immune_genes=[]

# for i in range(len(cglist)):
#     if (cglist[i] in igenes):
#         Tgeneinds.append(i)
#         fin_immune_genes.append(cglist[i])
        

# Tgeneinds=np.squeeze(np.asarray(Tgeneinds))

# Gi=Gr[:,Tgeneinds]


# D={}
# D['Xr']=Xr
# D['Gi']=Gi
# D['geneid']=geneid
# D['moneid']=moneid
# D['fin_immune_genes']=fin_immune_genes
# D['snum']=snum
# D['Tgeneinds']=Tgeneinds

# pickle.dump(D,open('LUADimmuneData.p','wb'))

###############################################################
####load data

D=pickle.load(open('LUADimmuneData.p','rb'))
Xr=D['Xr']
Gi=D['Gi']
geneid=D['geneid']
moneid=D['moneid']
fin_immune_genes=D['fin_immune_genes']
snum=D['snum']
Tgeneinds=D['Tgeneinds']


cmat=LedoitWolf().fit(np.concatenate((Xr,Gi),axis=1))
cmat=cmat.covariance_
vvec=np.squeeze(np.diagonal(cmat))
vvec=vvec[:,np.newaxis]
cmat=cmat/np.sqrt(vvec*vvec.T)

cmatr=cmat[:len(moneid),len(moneid):]

Fccmat=0.5*np.log((1+cmatr+T)/(1-cmatr+T))
Fccmat[np.abs(Fccmat)<Tc]=0
Fccmat[Fccmat>Tc]=Fccmat[Fccmat>Tc]-Tc
Fccmat[Fccmat<-Tc]=Fccmat[Fccmat<-Tc]+Tc
   
Zsc=np.squeeze(Fccmat)*np.sqrt(snum-3)   
Pv=2*stats.norm.sf(np.abs(Zsc))
Pv[np.isnan(Pv)]=T2
Pv[Pv<T2]=T2

Q=smt.multipletests(np.ravel(Pv),alpha=0.05, method='fdr_bh')
Q=np.squeeze(Q[1])
Nsig=np.sum(Q<0.05)
Qmat=np.reshape(Q,[len(moneid),len(Tgeneinds)])
Qmin=np.squeeze(np.min(Qmat,axis=1))

imoneinds=np.where(Qmin<0.05)
imones=moneid[imoneinds]
mnum=len(imones)

print('the number of significant mones is '+str(len(imones)))

#Xm=X[:,imones]

##################################################################
##################################################################
##create clustermap of corrs
sigM=np.sum(Qmat<0.05,axis=0)
#cmats=cmatr[np.squeeze(imoneinds),:]
cmats=cmatr[:,np.squeeze(np.where(sigM>0))]

#Qmats=Qmat[np.squeeze(imoneinds),:]
Qmats=Qmat[:,np.squeeze(np.where(sigM>0))]


sigG=np.sum(Qmat<0.05,axis=1)

cmats=cmats[np.squeeze(np.where(sigG>2)),:]
Qmats=Qmats[np.squeeze(np.where(sigG>2)),:]

mone_names=[]
for i in range(len(moneid)):
    if sigG[i]>2:
        mone_names.append('mone'+str(moneid[i]))

gene_names=[]
for i in range(len(sigM)):
    if sigM[i]>0:
        gene_names.append(fin_immune_genes[i])

plt.figure(dpi=1200)
sns.set(font_scale=2)
c=sns.clustermap(cmats,vmin=-0.5,vmax=0.5,cmap='seismic',yticklabels=mone_names,xticklabels=gene_names,annot_kws={"size": 32})
plt.savefig('luad_immune_MG.png',dpi=1200)

rot=c.dendrogram_row.reordered_ind
cot=c.dendrogram_col.reordered_ind

cmats_ord=cmats[rot,:]
cmats_ord=cmats_ord[:,cot]

Qmats_ord=Qmats[rot,:]
Qmats_ord=Qmats_ord[:,cot]

gene_names_ord=[]
mone_names_ord=[]

for i in range(len(gene_names)):
    gene_names_ord.append(gene_names[cot[i]])

for i in range(len(mone_names)):
    mone_names_ord.append(mone_names[rot[i]])


labs=[[' ' for i in range(len(gene_names))] for j in range(len(mone_names))]
for i in range(len(gene_names)):
    for j in range(len(mone_names)):
        if Qmats_ord[j,i]<0.05:
            labs[j][i]='*'
            
fig=plt.figure(dpi=1200)
ax = sns.heatmap(Qmats_ord, cmap="RdGy", vmin=0, vmax=0.1, annot=labs, annot_kws={'fontsize': 16, 'color' : 'b', 'size': 12}, fmt='s')
plt.xticks(0.5+np.arange(len(gene_names)),gene_names_ord,rotation=90,fontsize=8)
plt.yticks(0.5+np.arange(len(mone_names)),mone_names_ord, rotation=0,fontsize=8)
ax.grid(False)
#plt.show()
plt.savefig('luad_immune_MG_adj_pvals.png',dpi=1200)


