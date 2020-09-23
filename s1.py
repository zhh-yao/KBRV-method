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
def corr(row, col, t):
    
    xs = list(range(0, row, int(row/10)))
    ys = []
    
    for i in range(10):
        mat1 = np.array(np.random.normal(0, 1, row*col).reshape(row, col))
        mat2 = np.array(np.random.normal(0, 1, row*col).reshape(row, col))
        mat2 = sparsity(mat1, mat2, 0.1*i, t)
        
        r = RV2(mat1, mat2)
        ys.append(r)
        
    return xs, ys


# In[5]:

# plot 
fig, ax = plt.subplots(figsize=(12, 9))

xs, ys1 = corr(1000, 1000, 'l')
plt.plot(xs, ys1, lw=5, ls='--', label='linear')
xs, ys2 = corr(1000, 1000, 'nl')
plt.plot(xs, ys2, lw=5, label='nonlinear')
# ax.set_title('RV2 under nonlinear situation yij = xij^2\n', fontsize=30)
plt.xticks(xs, np.linspace(100, 10, 10))
plt.tick_params(axis='both', length=8, width=2, labelsize=20)
font = {'weight' : 'normal', 'size' : 35}
plt.xlabel('Sparsity %', font)
plt.ylabel('RV2 value', font)
plt.legend(loc='best', prop = font)

plt.savefig('s1.eps', bbox_inches='tight')
plt.savefig('s1.tif', bbox_inches='tight')
plt.show()



    











