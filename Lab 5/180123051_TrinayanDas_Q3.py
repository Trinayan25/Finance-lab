#!/usr/bin/env python
# coding: utf-8

# In[2]:


import statistics
import numpy as np
import pandas as pd
filepath=r"nse_index.csv"
b = pd.read_csv(filepath)
filepath2=r"bse_index.csv"
a = pd.read_csv(filepath2)
nse_ret,bse_ret=[],[]

for i in range(59):
  nse_ret.append( (b.loc[i+1,'nse_index']-b.loc[i,'nse_index'])/b.loc[i,'nse_index'] )
  bse_ret.append( (a.loc[i+1,'bse_index']-a.loc[i,'bse_index'])/a.loc[i,'bse_index'] )
variable_nse_ret,var_bseret=statistics.variance(nse_ret),statistics.variance(bse_ret)

  
def main(type): 
  s=['bse','nse','non_nse']
  t=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if type==0:
    filepathn=r"bsedata1.csv"
  if type==1:
    filepathn=r"nsedata1.csv"
  if type==2:
    filepathn=r"nse_non_index_data1.csv"
  print('\nBeta for',s[type],'stocks\n')
  defcon1 = pd.read_csv(filepathn)
  stocks=[]
  m=0
  for col in defcon1.columns:
    if m>0:
      stocks.append(col)
    m+=1
  n=len(stocks)
  for i in range(n):
    print("Name of the stock :",stocks[i])
    temp,tempret=[],[]
    for j in range(60):
      temp.append( defcon1.loc[j,stocks[i]])
      if j>0:
        tempret.append( (temp[j]-temp[j-1])/temp[j-1] )
    
    if type==1 or type==2:
       u=sum(nse_ret)/len(nse_ret)
    else :
       u=sum(bse_ret)/len(bse_ret)
    v,c=sum(tempret)/len(tempret),0
    if type==1 or type==2:
      for k in range(len(tempret)):
         c+=(tempret[k]-v)*(nse_ret[k]-u)
      c=c/(len(tempret))
      print("The value of beta for the stock :", c/variable_nse_ret,'\n\n')
    else :
      for k in range(len(tempret)):
         c+=(tempret[k]-v)*(bse_ret[k]-u)
      c=c/(len(tempret))
      print("The value of beta for the stock  :", c/var_bseret,'\n\n')
  print('\n\n',"************",'\n\n')

  

  
  
  
main(0)
main(1)
main(2)


# In[ ]:




