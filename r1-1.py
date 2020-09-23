# In[1]:

# import packages
import numpy as np
import pandas as pd
from minepy import MINE
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:

# select gene
def dataGenerate(path, picked):
    
    gene_all = pd.read_csv(path,sep='\t')

    geneData = gene_all.loc[gene_all.ID.isin(picked.index)].groupby('ID').sum()

    return geneData
    

# In[3]:

def normalize(data):
    
    r, c = np.shape(data)
    
    for i in range(r):
        
        rmax, rmin = max(data.iloc[i,:]), min(data.iloc[i,:])
        data.iloc[i,:] = (data.iloc[i,:] - rmin)/(rmax - rmin)
    
    return data


# In[4]:

# MIC method
def micMat(data):
    
    mine = MINE(alpha=0.6, c=15)
    corr_vec = np.zeros(shape=(9,9))
    for i in range(9):
        for j in range(9):
            mine.compute_score(np.array(data)[i,:], np.array(data)[j,:])
            corr_vec[i,j] = mine.mic()
    
    return pd.DataFrame(corr_vec, index=data.index , columns=data.index)


# In[5]:

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
    
    plt.savefig('r1.tif', bbox_inches='tight')
    plt.savefig('r1.eps', bbox_inches='tight')
    plt.show()


# In[6]:

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


# In[7]:

# mic
picked = pd.read_csv('FOXM1.txt',sep='\t').set_index('ID')
gene = dataGenerate('Gene.txt', picked)
gene = normalize(gene)

corr_mat = micMat(gene)
hmplot(corr_mat)

thre = threshold(corr_mat, cutoff=0.2)
corr_mat_01 = transform(corr_mat, thre)


















