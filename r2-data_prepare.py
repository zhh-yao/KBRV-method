# In[1]:

# import packages
import numpy as np
import pandas as pd


# In[2]:

# selcet and save exon-level data
def select_ensembl(Ensembl, g):
    
    key = Ensembl.loc[g,'name']
    
    return key
    

def select_exon(data, key):
    
    exon  = []
    
    for i in range(len(data)):
                
        if key in data.loc[i,'Ensembl']:
            e = data.loc[i,'Ensembl']
            exon.append(e)
    
    return exon


def select_count(data, key):
    
    count = []
    
    for i in range(len(data)):
                
        if key in data.loc[i,'Ensembl']:
            c = data.loc[i,'count']
            count.append(c)
    
    return count


def geneMat(cell_line, time, condition, key):
    
    data = pd.read_csv('%s_%dh_Rep%d_exon_read_counts.txt' %(cell_line, time[0], condition[0]), 
                       sep='\t', names=['Ensembl','count'])
    exon = select_exon(data, key)
    
    mat = pd.DataFrame(columns=exon)
    
    for i in time:
        for j in condition:
            data = pd.read_csv('%s_%dh_Rep%d_exon_read_counts.txt' %(cell_line, i, j), 
                               sep='\t', names=['Ensembl','count'])
            count = select_count(data, key)
            mat.loc[len(mat)] = count
    
    return mat


time = [3,6,12,24,48,96,120]
condition = [1,2,3]
Ensembl = pd.read_csv('Ensembl.txt', names=['name'])


## cell line = Macrophage
cell_line = 'Macrophage'

for i in range(len(Ensembl)):
    key = select_ensembl(Ensembl, i)
    exec( f'Mac_mat_%s=geneMat(cell_line, time, condition, key)' % i)
### save matrix
for i in range(len(Ensembl)):
    exec( f"Mac_mat_%s.to_csv('Mac_mat_%s.txt', sep='\t', index=False)" % (i, i))


## cell line = Monocyte
cell_line = 'Monocyte'

for i in range(len(Ensembl)):
    key = select_ensembl(Ensembl, i)
    exec( f'Mon_mat_%s=geneMat(cell_line, time, condition, key)' % i)
### save matrix
for i in range(len(Ensembl)):
    exec( f"Mon_mat_%s.to_csv('Mon_mat_%s.txt', sep='\t', index=False)" % (i, i))


## cell line = Neutrophil
cell_line = 'Neutrophil'

for i in range(len(Ensembl)):
    key = select_ensembl(Ensembl, i)
    exec( f'Neu_mat_%s=geneMat(cell_line, time, condition, key)' % i)
### save matrix
for i in range(len(Ensembl)):
    exec( f"Neu_mat_%s.to_csv('Neu_mat_%s.txt', sep='\t', index=False)" % (i, i))


# In[3]:

# selcet and save rna-seq data
def gene_count(data, key):
        
    for i in range(len(data)):
        
        if key in data.loc[i,'Ensembl']:
            count = data.loc[i,'count']
            
    return count


def geneVec(cell_line, time, condition, key):
    
    gene = []
    
    for i in time:
        for j in condition:
            data = pd.read_csv('%dh-%s-Rep%d.txt' %(i, cell_line, j),
                               skiprows=1, sep='\t', names=['Ensembl','count'])
            gene.append(gene_count(data, key))
            
    return np.array(gene)


## cell line = Macrophage
cell_line = 'Mac'

for i in range(len(Ensembl)):
    key = select_ensembl(Ensembl, i)
    exec( f'Mac_vec_%s=geneVec(cell_line, time, condition, key)' % i)
### save vector
for i in range(len(Ensembl)):
    exec( f"np.savetxt('Mac_vec_%s.txt', Mac_vec_%s)" % (i, i))

## cell line = Monocyte
cell_line = 'Mon'

for i in range(len(Ensembl)):
    key = select_ensembl(Ensembl, i)
    exec( f'Mon_vec_%s=geneVec(cell_line, time, condition, key)' % i)
### save vector
for i in range(len(Ensembl)):
    exec( f"np.savetxt('Mon_vec_%s.txt', Mon_vec_%s)" % (i, i))

## cell line = Neutrophil
cell_line = 'Neu'

for i in range(len(Ensembl)):
    key = select_ensembl(Ensembl, i)
    exec( f'Neu_vec_%s=geneVec(cell_line, time, condition, key)' % i)
### save vector
for i in range(len(Ensembl)):
    exec( f"np.savetxt('Neu_vec_%s.txt', Neu_vec_%s)" % (i, i))







