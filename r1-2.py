# In[1]:

# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:

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


# In[3] :

# permutation test
def permutation(data1, data2, alpha, per):
    
    count = 0
    r, c = data1.shape
    kbrv = KBRV(data1, data2, alpha)
    
    for i in range(per):
        
        temp = data1.reshape(1, r*c)[0,:]
        data1_ = np.random.permutation(temp).reshape(r, c)
        rand = KBRV(data1_, data2, alpha)
        if rand >= kbrv:
            count += 1
    
    p = count/per
    
    return p


# In[4]:

# select gene
def dataGenerate(picked):
    
    gene_all = pd.read_csv('Gene.txt',sep='\t')
    meth_all = pd.read_csv('Methylation.txt',sep='\t')

    geneData = gene_all.loc[gene_all.ID.isin(picked.index)].groupby('ID').sum()
    methData = meth_all.loc[meth_all.ID.isin(picked.index)].groupby('ID').sum()

    return geneData, methData
    

# In[5]:

def normalize(data):
    
    r, c = np.shape(data)
    
    for i in range(r):
        
        rmax, rmin = max(data.iloc[i,:]), min(data.iloc[i,:])
        data.iloc[i,:] = (data.iloc[i,:] - rmin)/(rmax - rmin)
    
    return data


# In[6]:

# KBRV method
def kbrvMat(gene, meth):
    
    corr_mat  = np.zeros(shape=(9,9))
    alpha_mat = np.zeros(shape=(9,9))
    alpha = np.arange(0, 1.01, 0.01)
    temp = []
    alpha_ = []
    
    for i in range(1,9):
        for j in range(i):
            data1 = np.array([np.array(gene)[i,:], np.array(meth)[i,:]]).T
            data2 = np.array([np.array(gene)[j,:], np.array(meth)[j,:]]).T
            for k in alpha:
                if permutation(data1, data2, k, 100) < 0.05:
                    temp.append(KBRV(data1, data2, k))
                    alpha_.append(k)
            if len(temp) != 0:
                corr_mat[i,j]  = max(temp)
                alpha_mat[i,j] = alpha_[temp.index(max(temp))]
            else:
                corr_mat[i,j] = 0
                alpha_mat[i,j] = -1
            temp = []
    
    return pd.DataFrame(corr_mat, index=gene.index , columns=gene.index), pd.DataFrame(alpha_mat, index=gene.index , columns=gene.index)


def diaMat(data):
    
    l = len(data)
    
    for i in range(l):
        for j in range(i,l):
            if i == j:
                data.iloc[i,j] = 1
            else:
                data.iloc[i,j] = data.iloc[j,i]
    
    return data


# In[7]:

# plot heatmap
def hmplot(data):
    
    f, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(data, vmax=1, square=True, cmap="YlGnBu", linewidths=1)
    
    ax.set_xlabel('OFFICIAL GENE SYMBOL', fontsize=30)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, rotation=30)
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)
    
    ax.tick_params(bottom=False,top=False,left=False,right=False)
    
    plt.savefig('r2.tif', bbox_inches='tight')
    plt.savefig('r2.eps', bbox_inches='tight')
    plt.show()

def hmplot2(data):
    
    f, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(data, annot=True, vmax=1, square=True, cmap="Blues", linewidths=1, cbar=False, annot_kws={'size':20, 'weight':'bold', 'color':'lightcoral'})
    
    ax.set_xlabel('OFFICIAL GENE SYMBOL', fontsize=30)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, rotation=30)
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)
    
    ax.tick_params(bottom=False,top=False,left=False,right=False)
    
    plt.savefig('r2_101.tif', bbox_inches='tight')
    plt.savefig('r2_101.eps', bbox_inches='tight')
    plt.show()


# In[8]:

# threshold
def threshold(data, cutoff=0.1):
    
    l = len(data)
    temp = []
    
    for i in range(l):
        for j in range(i+1,l):
            temp.append(data.iloc[i,j])
            
    thre = np.percentile(temp, (1-cutoff)*100)
    
    return thre


def transform(data, thre):
    
    new = pd.DataFrame(columns=data.columns, index=data.columns)
    l = len(data)
    
    for i in range(l):
        for j in range(l):
            if data.iloc[i,j] < thre:
                new.iloc[i,j] = 0
            else:
                new.iloc[i,j] = 1
    
    return new


def trans2(data_01, data):
    
    l = len(data)
    
    for i in range(l):
        for j in range(l):
            if data_01.iloc[i,j] == 0:
                data.iloc[i,j] = -1
    
    return data


# In[9]:

# kbrv
picked = pd.read_csv('FOXM1.txt',sep='\t').set_index('ID')
gene, meth = dataGenerate(picked)
gene = normalize(gene)
# you can obtain corr_mat and corr_mat_alpha through the following steps
corr_mat, corr_mat_alpha = kbrvMat(gene, meth)
corr_mat, corr_mat_alpha = diaMat(corr_mat), diaMat(corr_mat_alpha)
corr_mat.to_excel('corr_mat.xlsx')
corr_mat_alpha.to_excel('corr_mat_alpha.xlsx')
# or you can obtain them from our saved csv file
'''
corr_mat = pd.read_excel('corr_mat.xlsx', index_col=0)
corr_mat_alpha = pd.read_excel('corr_mat.xlsx', index_col=0)
'''

hmplot(corr_mat)

thre = threshold(corr_mat, cutoff=0.2)
corr_mat_01 = transform(corr_mat, thre)
corr_mat_alpha_101 = trans2(corr_mat_01, corr_mat_alpha)

hmplot2(corr_mat_alpha_101)





















