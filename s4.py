# In[1]:

# import packages
import numpy as np
import random
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


# In[3]:

# define sparsity
def sparsity(data1, data2, percent, t):
    
    r,l = data1.shape
    ind = list(range(0,l))
    
    for i in ind:
        indr = random.sample(ind, int(l*percent))
        for j in ind:
            if j in indr:
                if t == 'l':
                    data2[i,j] = data1[i,j]
                elif t == 'nl':
                    data2[i,j] = data1[i,j]**2
            else:
                data2[i,j] = random.gauss(0,1)
    
    return data2
    

# In[4]:

# calculate linear/nonlinear relationship
def corr(row, col, t, alpha):
    
    xs = list(range(0, row, int(row/10)))
    ys = []
    
    for i in range(10):
        mat1 = np.array(np.random.normal(0, 1, row*col).reshape(row, col))
        mat2 = np.array(np.random.normal(0, 1, row*col).reshape(row, col))
        mat2 = sparsity(mat1, mat2, 0.1*i, t)
        
        r = KBRV(mat1, mat2, alpha)
        ys.append(r)
        
    return xs, ys


# In[5-1]:

# plot: linear
fig, ax = plt.subplots(figsize=(12, 9))

xs, ys1 = corr(1000, 1000, 'l', 0)
xs, ys2 = corr(1000, 1000, 'l', 0.3)
xs, ys3 = corr(1000, 1000, 'l', 0.5)
xs, ys4 = corr(1000, 1000, 'l', 0.7)
xs, ys5 = corr(1000, 1000, 'l', 1)

plt.plot(xs, ys1, lw=5, label='alpha=0')
plt.plot(xs, ys2, lw=5, marker='o', markersize=13, label='alpha=0.3')
plt.plot(xs, ys3, lw=5, marker='+', markersize=15, label='alpha=0.5')
plt.plot(xs, ys4, lw=5, marker='*', markersize=15, label='alpha=0.7')
plt.plot(xs, ys5, lw=5, ls='--', label='alpha=1')

plt.xticks(xs, np.linspace(100, 10, 10))
plt.tick_params(axis='both', length=8, width=2, labelsize=20)
font1 = {'weight' : 'normal', 'size' : 35}
plt.xlabel('Sparsity %', font1)
plt.ylabel('KBRV value', font1)
font2 = {'weight' : 'normal', 'size' : 30}
plt.legend(loc='best', prop = font2)

plt.savefig('s4-1.eps', bbox_inches='tight')
plt.savefig('s4-1.tif', bbox_inches='tight')
plt.show()


# In[5-2]:

# plot: nonlinear
fig, ax = plt.subplots(figsize=(12, 9))

xs, ys1 = corr(1000, 1000, 'nl', 0)
xs, ys2 = corr(1000, 1000, 'nl', 0.3)
xs, ys3 = corr(1000, 1000, 'nl', 0.5)
xs, ys4 = corr(1000, 1000, 'nl', 0.7)
xs, ys5 = corr(1000, 1000, 'nl', 1)

plt.plot(xs, ys1, lw=5, label='alpha=0')
plt.plot(xs, ys2, lw=5, marker='o', markersize=13, label='alpha=0.3')
plt.plot(xs, ys3, lw=5, marker='+', markersize=15, label='alpha=0.5')
plt.plot(xs, ys4, lw=5, marker='*', markersize=15, label='alpha=0.7')
plt.plot(xs, ys5, lw=5, ls='--', label='alpha=1')

plt.xticks(xs, np.linspace(100, 10, 10))
plt.tick_params(axis='both', length=8, width=2, labelsize=20)
font1 = {'weight' : 'normal', 'size' : 35}
plt.xlabel('Sparsity %', font1)
plt.ylabel('KBRV value', font1)
font2 = {'weight' : 'normal', 'size' : 30}
plt.legend(loc='best', prop = font2)

plt.savefig('s4-2.eps', bbox_inches='tight')
plt.savefig('s4-2.tif', bbox_inches='tight')
plt.show()














