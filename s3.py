# In[1]:

# import packages
import numpy as np
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

# calculate correlations with different samples
def corr(alpha):
    
    xs = list(range(50, 1000, 50))
    ys = []
    
    for i in xs:
        
        kbrv = []
        for j in range(100):
            x = np.array(np.random.normal(0, 1, i*500).reshape(i, 500))
            y = np.array(np.random.normal(0, 1, i*800).reshape(i, 800))
            kbrv.append(KBRV(x, y, alpha))
        
        ys.append(np.mean(kbrv))
            
    return ys
    

# In[5]:

# plot 
fig, ax = plt.subplots(figsize=(12, 9))

xs = list(range(50, 1000, 50))
ys1 = corr(0)
ys2 = corr(0.3)
ys3 = corr(0.5)
ys4 = corr(0.7)
ys5 = corr(1)

plt.plot(xs, ys1, lw=5, label='alpha=0')
plt.plot(xs, ys2, lw=5, marker='o', markersize=13, label='alpha=0.3')
plt.plot(xs, ys3, lw=5, marker='+', markersize=13, label='alpha=0.5')
plt.plot(xs, ys4, lw=5, marker='*', markersize=13, label='alpha=0.7')
plt.plot(xs, ys5, lw=5, ls='--', label='alpha=1')
# ax.set_title('RV2 under nonlinear situation yij = xij^2\n', fontsize=30)
plt.tick_params(axis='both', length=8, width=2, labelsize=20)
font1 = {'weight' : 'normal', 'size' : 35}
plt.xlabel('Sample size', font1)
plt.ylabel('KBRV value', font1)
font2 = {'weight' : 'normal', 'size' : 30}
plt.legend(loc='best', prop = font2)

plt.savefig('s3.eps', bbox_inches='tight')
plt.savefig('s3.tif', bbox_inches='tight')
plt.show()














