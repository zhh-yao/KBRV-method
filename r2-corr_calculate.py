# In[1]:

# import packages
import numpy as np
import pandas as pd
from minepy import MINE
import warnings


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

# import data and calculate correlation
def vec_corr(Symble, cell_line, alpha):
    
    mat = pd.DataFrame(columns=Symble['name'])
    l = len(Symble)

    mine = MINE(alpha, c=15)
    
    for i in range(l):
        corr = []
        for j in range(l):
            exec( f"mine.compute_score(np.log2(%s_vec_%s+1), np.log2(%s_vec_%s+1))" % (cell_line, i, cell_line, j))
            corr.append(mine.mic())
        mat.loc[len(mat)] = corr

    return mat
    

def kbrv_optimal(data1, data2):
    
    kbrvs = []
    alphas = []
    
    for alpha in np.arange(0, 1.1, 0.1):
        if permutation(data1, data2, alpha, 100) < 0.05:
            kbrvs.append(KBRV(data1, data2, alpha))
            alphas.append(alpha)
    if len(kbrvs) != 0:
        bestK = max(kbrvs)
        bestA = max(alphas)
    else:
        bestK = 0
        bestA = -1
    
    return bestK, bestA
    

def mat_corr(Symble, cell_line):
    
    mat_corr = pd.DataFrame(columns=Symble['name'], index=Symble['name'])
    mat_alpha = pd.DataFrame(columns=Symble['name'], index=Symble['name'])
    
    l = len(Symble)
    
    for i in range(1, l):
        for j in range(i):

            exec( f"mat_corr.iloc[%d,%d], mat_alpha.iloc[%d,%d] = kbrv_optimal(np.log2(np.array(%s_mat_%s+1)), np.log2(np.array(%s_mat_%s+1)))" % (i,j,i,j,cell_line, i, cell_line, j))
    
    return mat_corr, mat_alpha


def diaMat(data):
    
    l = len(data)
    
    for i in range(l):
        for j in range(i,l):
            if i == j:
                data.iloc[i,j] = 1
            else:
                data.iloc[i,j] = data.iloc[j,i]
    
    data = data.fillna(0)
    
    return data


# In[5]:

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


# In[6]:

# import data
Symble = pd.read_csv('Symble.txt', names=['name'])
for i in range(len(Symble)):
    
    exec( f"Mac_vec_%s = np.loadtxt('Mac_vec_%s.txt')" % (i, i))
    exec( f"Mon_vec_%s = np.loadtxt('Mon_vec_%s.txt')" % (i, i))
    exec( f"Neu_vec_%s = np.loadtxt('Neu_vec_%s.txt')" % (i, i))

    exec( f"Mac_mat_%s = pd.read_csv('Mac_mat_%s.txt', sep='\t')" % (i, i))
    exec( f"Mon_mat_%s = pd.read_csv('Mon_mat_%s.txt', sep='\t')" % (i, i))
    exec( f"Neu_mat_%s = pd.read_csv('Neu_mat_%s.txt', sep='\t')" % (i, i))


# calculate correlation by MIC
warnings.filterwarnings('ignore')

alpha = 0.6
corr_Mac_vec = vec_corr(Symble, 'Mac', alpha)
corr_Mon_vec = vec_corr(Symble, 'Mon', alpha)
corr_Neu_vec = vec_corr(Symble, 'Neu', alpha)

## Mac_vec
thre = threshold(corr_Mac_vec, cutoff=0.1)
corr_Mac_vec_01 = transform(corr_Mac_vec, thre)
## Mon_vec
thre = threshold(corr_Mon_vec, cutoff=0.1)
corr_Mon_vec_01 = transform(corr_Mon_vec, thre)
## Neu_vec
thre = threshold(corr_Neu_vec, cutoff=0.1)
corr_Neu_vec_01 = transform(corr_Neu_vec, thre)


# In[7]:

# calculate correlation and alpha by KBRV
corr_Mac_mat, alpha_Mac = mat_corr(Symble, 'Mac')
corr_Mon_mat, alpha_Mon = mat_corr(Symble, 'Mon')
corr_Neu_mat, alpha_Neu = mat_corr(Symble, 'Neu')

corr_Mac_mat, corr_Mon_mat, corr_Neu_mat = diaMat(corr_Mac_mat), diaMat(corr_Mon_mat), diaMat(corr_Neu_mat)
alpha_Mac, alpha_Mon, alpha_Neu = diaMat(alpha_Mac), diaMat(alpha_Mon), diaMat(alpha_Neu)

corr_Mac_mat.to_excel('corr_Mac_mat.xlsx')
corr_Mon_mat.to_excel('corr_Mon_mat.xlsx')
corr_Neu_mat.to_excel('corr_Neu_mat.xlsx')
alpha_Mac.to_excel('alpha_Mac.xlsx')
alpha_Mon.to_excel('alpha_Mon.xlsx')
alpha_Neu.to_excel('alpha_Neu.xlsx')

# or you can obtain them from our saved csv file
'''
corr_Mac_mat = pd.read_excel('corr_Mac_mat.xlsx', index_col=0)
corr_Mon_mat = pd.read_excel('corr_Mon_mat.xlsx', index_col=0)
corr_Neu_mat = pd.read_excel('corr_Neu_mat.xlsx', index_col=0)
alpha_Mac = pd.read_excel('alpha_Mac.xlsx', index_col=0)
alpha_Mon = pd.read_excel('alpha_Mon.xlsx', index_col=0)
alpha_Neu = pd.read_excel('alpha_Neu.xlsx', index_col=0)
'''

## Mac_mat
thre = threshold(corr_Mac_mat, cutoff=0.1)
corr_Mac_mat = transform(corr_Mac_mat, thre)
## Mon_mat
thre = threshold(corr_Mon_mat, cutoff=0.1)
corr_Mon_mat = transform(corr_Mon_mat, thre)
## Neu_mat
thre = threshold(corr_Neu_mat, cutoff=0.1)
corr_Neu_mat = transform(corr_Neu_mat, thre)

alpha_Mac_101 = trans2(corr_Mac_mat, alpha_Mac)
alpha_Mon_101 = trans2(corr_Mon_mat, alpha_Mon)
alpha_Neu_101 = trans2(corr_Neu_mat, alpha_Neu)





