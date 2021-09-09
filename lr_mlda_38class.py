import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression as LR
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import random



#masird='/projects/chuang-lab/USERS/foroua/ML2/preproc/medsFFPE_NPT/'
#masirw='/projects/chuang-lab/USERS/foroua/ML2/DSL_final/'

masird='./down_samp_cans/'
masirw='./'

fdim=2048
rho=0.7  ###train test split ratio
Cvec=[10]  ###the C values for cross classification
Cvec38=[10]  ###the c values for the 38 class LR-LASSO classifier
MI=100  ##max iteration, the defualt value of sklearn, used for being clear

cancerjavad=['lihc','thca','paad','kirp','ov','hnsc','brca','esca','sarc','blca','read','stad','coad','prad','ucec','luad','lusc','kich','kirc']
cancerjavad.sort()  ###the list of cancers analyzed in alphabetical order
cnum=len(cancerjavad)

dT=30000


for runnum in range(10):
    
    ######################
    ###start here


    
    print('started run id '+str(runnum))

    ###load the data and split to train and test
    trainD={}
    testD={}
    for cancer in cancerjavad:
        
        W=pickle.load( open( masird+cancer+".p", "rb" ) )
        feats=W['mfeats']
        y=W['lab']
        sname=W['sname']
        sbar=W['sbar']
        isffpe=W['isffpe']
        pid=[]
        spid=[]
        for csbar in sbar:
            pid.append(csbar[0:12])
            spid.append(csbar[0:12])
        
        pid=list(set(pid))
        random.shuffle(pid)
        
        pid_train=[]
        pid_test=[]
        
        for i in range(len(pid)):
            
            cpid=pid[i]
            
            if i<rho*len(pid):
                pid_train.append(cpid)
            else:
                pid_test.append(cpid)
        
        snum=len(y)
        cX_train=np.zeros((snum,fdim))
        cX_test=np.zeros((snum,fdim))
        
        cY_train=np.zeros((snum,))
        cY_test=np.zeros((snum,))
        
        cntt=0
        cnts=0
        for i in range(snum):
            
            cpid=spid[i]
            cy=y[i]
            cfeats=feats[i,:]
            cisffpe=isffpe[i]
            
            if (cpid in pid_train) and (cisffpe<1):
                    
                cX_train[cntt,:]=cfeats
                cY_train[cntt]=cy
                cntt=cntt+1
                
            elif (cpid in pid_test) and (cisffpe<1):
                
                cX_test[cnts,:]=cfeats
                cY_test[cnts]=cy
                cnts=cnts+1
            
        cX_train=cX_train[:cntt,:]
        cY_train=cY_train[:cntt]
        cX_test=cX_test[:cnts,:]
        cY_test=cY_test[:cnts]
        
        trainD[cancer]=[cX_train,cY_train]
        testD[cancer]=[cX_test,cY_test]
        
        
        
    ##############################
    ###LR and MLDA the 38 class problem+UTN!
    Xtrain=np.zeros((dT,fdim))
    Ytrain=np.zeros((dT,))
    Ztrain=np.zeros((dT,))
    Ttrain=np.zeros((dT,))
    Xtest=np.zeros((dT,fdim))
    Ytest=np.zeros((dT,))
    Ztest=np.zeros((dT,))
    Ttest=np.zeros((dT,))
    
    cancnt=0
    cntt=0
    cnts=0
    for cancer in cancerjavad:
        cXtrain, cYtrain=trainD[cancer]
        snum=len(cYtrain)
        Xtrain[cntt:cntt+snum,:]=cXtrain
        Ytrain[cntt:cntt+snum]=cYtrain
        Ztrain[cntt:cntt+snum]=2*cancnt+cYtrain
        Ttrain[cntt:cntt+snum]=cancnt
        cntt=cntt+snum
        
        cXtest, cYtest=testD[cancer]
        snum=len(cYtest)
        Xtest[cnts:cnts+snum,:]=cXtest
        Ytest[cnts:cnts+snum]=cYtest
        Ztest[cnts:cnts+snum]=2*cancnt+cYtest
        Ttest[cnts:cnts+snum]=cancnt
        cnts=cnts+snum
        
        cancnt=cancnt+1
    
    Xtrain=Xtrain[:cntt,:]
    Ytrain=Ytrain[:cntt]
    Ztrain=Ztrain[:cntt]
    Ttrain=Ttrain[:cntt]
    
    Xtest=Xtest[:cnts,:]
    Ytest=Ytest[:cnts]
    Ztest=Ztest[:cnts]
    Ttest=Ttest[:cnts]
        
    
    ###first do 38 class MLDA
    aucMLDA38={}
    lda38=LDA().fit(Xtrain,Ztrain)
    preds38=lda38.predict_proba(Xtest)
    
    for i in range(38):
        
        if (i % 2)==0:
            classid=cancerjavad[int(i/2)]+'_nomal'
        else:
            classid=cancerjavad[int(i/2)]+'_tumor'
        
        cZtest=np.squeeze(1*(Ztest==i))
        cpreds=np.squeeze(preds38[:,i])
        fpr, tpr, thresholds = metrics.roc_curve(cZtest, cpreds, pos_label=1)
        aucMLDA38[classid]=metrics.auc(fpr, tpr)
    
    #####train the 38class LR
    cm38={}
    aucLR38={}
    aucLRtissue={}
    aucLRtn38={}
    for cc in Cvec38:
        lr38=LR(penalty='l1',tol=1e-4,C=cc,solver='saga',max_iter=MI,class_weight='balanced',multi_class='multinomial').fit(Xtrain,Ztrain)
        preds38=lr38.predict_proba(Xtest)
        
        #####all 38 classes
        for i in range(38):
            
            if (i % 2)==0:
                classid=cancerjavad[int(i/2)]+'_nomal'
            else:
                classid=cancerjavad[int(i/2)]+'_tumor'
        
            cZtest=np.squeeze(1*(Ztest==i))
            cpreds=np.squeeze(preds38[:,i])
            fpr, tpr, thresholds = metrics.roc_curve(cZtest, cpreds, pos_label=1)
            aucLR38[classid+'_cc'+str(cc)]=metrics.auc(fpr, tpr)
        
        ####tissue of origin
        for i in range(19):
        
            cTtest=1*(Ttest==i)
            cpreds=np.squeeze(preds38[:,2*i]+preds38[:,2*i+1])
            fpr, tpr, thresholds = metrics.roc_curve(cTtest, cpreds, pos_label=1)
            aucLRtissue[cancerjavad[i]+str(cc)]=metrics.auc(fpr, tpr)
        
        ####TN status from 38 classes:
        cpreds=np.squeeze(np.sum(preds38[:,1::2],axis=1))
        fpr, tpr, thresholds = metrics.roc_curve(Ytest, cpreds, pos_label=1)
        aucLRtn38[str(cc)]=metrics.auc(fpr, tpr)
        
        
        ####make confusion matrix
        labs=np.argmax(preds38,axis=1)
        cm38['lr38cc'+str(cc)]=metrics.confusion_matrix(Ztest,labs)
        
    
    
    
    #######now CC LR
    print('starting CC')
    aucCC={}
    for cc in Cvec:
        for cancertrain in cancerjavad:
            
            cXtrain, cYtrain=trainD[cancertrain]
            
            lrcc=LR(penalty='l1',tol=1e-5,C=cc,solver='liblinear',max_iter=MI,class_weight='balanced').fit(cXtrain,cYtrain)
            
            for cancertest in cancerjavad:
                
                cXtest, cYtest=testD[cancertest]
                predsu=lrcc.predict_proba(cXtest)
                fpr, tpr, thresholds = metrics.roc_curve(cYtest, predsu[:,1], pos_label=1)
                aucCC[cancertrain+cancertest+str(cc)]=metrics.auc(fpr, tpr)
                
    
    
    ##############################################################
    #####save the results and save to disk
    W={}
    W['cm38']=cm38
    W['aucMLDA38']=aucMLDA38
    W['aucLR38']=aucLR38
    W['aucLRtissue']=aucLRtissue
    W['aucLRtn38']=aucLRtn38
    W['aucCC']=aucCC
    
    pickle.dump( W, open( masirw+"MLDA_LR_runF"+str(runnum)+"_down_samp.p", "wb" ) )
    
    