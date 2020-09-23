# In[1]:

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings


# In[2]:

# generate dependent matrix pairs
def dataGenerate_dep_linear(num1_exon, num2_exon, S, co):

    '''
    num1_exon: number of exon in gene 1;
    num2_exon: number of exon in gene 2;
    S: sample size;
    co: association strength;
    '''
    mu = np.zeros(num1_exon + num2_exon)
    SIGMA = np.eye(num1_exon + num2_exon)
    Simulation_isoform_matrix = np.random.multivariate_normal(mu, SIGMA, S)
    isoform1_expression = Simulation_isoform_matrix[:,:num1_exon]
    isoform2_expression = Simulation_isoform_matrix[:,num1_exon:]
    
    if num1_exon >= num2_exon:
        isoform1_expression = isoform1_expression + co*(isoform1_expression[:,0:num1_exon])
        isoform2_expression = isoform2_expression + co*(isoform1_expression[:,0:num2_exon])
    else:
        isoform1_expression = isoform1_expression + co*(isoform2_expression[:,0:num1_exon])
        isoform2_expression = isoform2_expression + co*(isoform2_expression[:,0:num2_exon])
    
    return isoform1_expression, isoform2_expression



# In[3]:

# generate independent matrix pairs
def dataGenerate_dep_nonlinear(num1_exon, num2_exon, S, co):

    mu = np.zeros(num1_exon + num2_exon)
    SIGMA = np.eye(num1_exon + num2_exon)
    Simulation_isoform_matrix = np.random.multivariate_normal(mu, SIGMA, S)
    isoform1_expression = Simulation_isoform_matrix[:,:num1_exon]
    isoform2_expression = Simulation_isoform_matrix[:,num1_exon:]
    
    if num1_exon >= num2_exon:
        isoform1_expression = isoform1_expression + co*(isoform1_expression[:,0:num1_exon])
        isoform2_expression = isoform2_expression + co*np.exp(isoform1_expression[:,0:num2_exon])
    else:
        isoform1_expression = isoform1_expression + co*(isoform2_expression[:,0:num1_exon])
        isoform2_expression = isoform2_expression + co*np.exp(isoform2_expression[:,0:num2_exon])
    
    return isoform1_expression, isoform2_expression


# In[4]:

# RV2
def RV2(data1, data2):
    s = data1.dot(data1.T)
    t = data2.dot(data2.T)
    i = s.shape[0]
    for k in range(i):
        s.itemset((k,k),0)
        t.itemset((k,k),0)
    rv2 = ((s*t).sum())/np.sqrt(((s*s).sum())*((t*t).sum()))
    return rv2

# RVa
def RVa(data1, data2):
    fData1 = 1/(1+np.exp(-data1))
    s = fData1.dot(fData1.T)
    t = data2.dot(data2.T)
    i = s.shape[0]
    for k in range(i):
        s.itemset((k,k),0)
        t.itemset((k,k),0)
    rva = ((s*t).sum())/np.sqrt(((s*s).sum())*((t*t).sum()))
    return rva

# RVb
def RVb(data1, data2):
    fData2 = 1/(1+np.exp(-data2))
    s = data1.dot(data1.T)
    t = fData2.dot(fData2.T)
    i = s.shape[0]
    for k in range(i):
        s.itemset((k,k),0)
        t.itemset((k,k),0)
    rvb = ((s*t).sum())/np.sqrt(((s*s).sum())*((t*t).sum()))
    return rvb

# KBRV
def KBRV(data1, data2, alpha):
    kbrv = alpha*RV2(data1,data2) + (1-alpha)*0.5*(RVa(data1,data2) + RVb(data1,data2))
    return kbrv


# In[5]:

# some parameter
number = [[50,50],[100,100],[50,200]]
sample = [50, 100, 200]
c = [0.1, 0.2, 0.3]
alpha = [0, 0.3, 0.5, 0.7, 1]

warnings.filterwarnings('ignore')


# In[6]:

# generate simulation data
print("calculating correlation coefficients, please waiting...")
for i in range(len(number)):
    for j in range(len(sample)):
        for k in range(len(c)):
            
            label = []
            kbrv1 = []
            kbrv2 = []
            kbrv3 = []
            kbrv4 = []
            kbrv5 = []

            for n in range(1000):

                label.append(1)
                data1, data2 = dataGenerate_dep_nonlinear(number[i][0], number[i][1], sample[j], c[k])
                kbrv1.append(abs(KBRV(data1, data2, alpha[0])))
                kbrv2.append(abs(KBRV(data1, data2, alpha[1])))
                kbrv3.append(abs(KBRV(data1, data2, alpha[2])))
                kbrv4.append(abs(KBRV(data1, data2, alpha[3])))
                kbrv5.append(abs(KBRV(data1, data2, alpha[4])))
                
                label.append(0)
                data1_, data2_ = dataGenerate_dep_linear(number[i][0], number[i][1], sample[j], c[k])
                kbrv1.append(abs(KBRV(data1_, data2_, alpha[0])))
                kbrv2.append(abs(KBRV(data1_, data2_, alpha[1])))
                kbrv3.append(abs(KBRV(data1_, data2_, alpha[2])))
                kbrv4.append(abs(KBRV(data1_, data2_, alpha[3])))
                kbrv5.append(abs(KBRV(data1_, data2_, alpha[4])))
                
            DataSet = list(zip(label,kbrv1,kbrv2,kbrv3,kbrv4,kbrv5))
            df = pd.DataFrame(data = DataSet, columns=['Label', 'KBRV1', 'KBRV2', 'KBRV3', 'KBRV4', 'KBRV5'])
            df.to_csv('D:/csv_lnl2/combination%d-%d-%d.csv'%(i,j,k), index = False, header = True)

print("calculation of correlation coefficients has been completed...")


# In[7]:

# calculate and plot ROC/AUC
print("calculating auc value and drawing roc curve, please waiting...")
for i in range(len(number)):
    for j in range(len(sample)):
        for k in range(len(c)):
            data = pd.read_csv('D:/csv_lnl2/combination%d-%d-%d.csv'%(i,j,k))
            fpr1, tpr1, threshold1 = roc_curve(data['Label'], data['KBRV1'])
            roc_auc1 = auc(fpr1,tpr1)
            fpr2, tpr2, threshold2 = roc_curve(data['Label'], data['KBRV2'])
            roc_auc2 = auc(fpr2,tpr2)
            fpr3, tpr3, threshold3 = roc_curve(data['Label'], data['KBRV3'])
            roc_auc3 = auc(fpr3,tpr3)
            fpr4, tpr4, threshold4 = roc_curve(data['Label'], data['KBRV4'])
            roc_auc4 = auc(fpr4,tpr4)
            fpr5, tpr5, threshold5 = roc_curve(data['Label'], data['KBRV5'])
            roc_auc5 = auc(fpr5,tpr5)
            
            # plot
            plt.figure()
            plt.figure(figsize=(15,13))

            plt.plot(fpr1, tpr1, lw=5, label='alpha=0(AUC=%0.2f)' % roc_auc1) 
            plt.plot(fpr2, tpr2, lw=5, marker='o', markersize=17,  markevery = 0.05, label='alpha=0.3(AUC=%0.2f)' % roc_auc2)
            plt.plot(fpr3, tpr3, lw=5, marker='+', markersize=20,  markevery = 0.05, label='alpha=0.5(AUC=%0.2f)' % roc_auc3)
            plt.plot(fpr4, tpr4, lw=5, marker='*', markersize=20,  markevery = 0.05, label='alpha=0.7(AUC=%0.2f)' % roc_auc4)
            plt.plot(fpr5, tpr5, lw=5, ls='--', label='alpha=1(AUC=%0.2f)' % roc_auc5)

            plt.plot([0, 1], [0, 1], color='gray')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.tick_params(axis='both', length=12, width=4, labelsize=30)

            font1 = {'weight' : 'normal', 'size' : 35}
            plt.xlabel('False Positive Rate', font1)
            plt.ylabel('True Positive Rate', font1)

            font2 = {'weight' : 'normal', 'size' : 30}
            plt.legend(loc='lower right', prop = font2)

            plt.savefig('D:/fig/%d+%d+%d+%0.2f.eps'%(number[i][0], number[i][1], sample[j], c[k]), bbox_inches='tight')
            plt.savefig('D:/fig/%d+%d+%d+%0.2f.tif'%(number[i][0], number[i][1], sample[j], c[k]), bbox_inches='tight')

print("calculation and drawing have been completed...")


