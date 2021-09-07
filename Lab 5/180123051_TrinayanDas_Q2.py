#!/usr/bin/env python
# coding: utf-8

# In[4]:


import statistics
import math
import matplotlib.pyplot as plt
import pandas as pd

filepath1=r"nse_index.csv"
b = pd.read_csv(filepath1)
filepath2=r"bse_index.csv"
a = pd.read_csv(filepath2)
nse_ret,bse_ret=[],[]

for i in range(59):
  nse_ret.append( (b.loc[i+1,'nse_index']-b.loc[i,'nse_index'])/b.loc[i,'nse_index'] )
  bse_ret.append( (a.loc[i+1,'bse_index']-a.loc[i,'bse_index'])/a.loc[i,'bse_index'] )
var_nseret,var_bseret=statistics.variance(nse_ret),statistics.variance(bse_ret)
sd_nse,sd_bse=math.sqrt(var_nseret),math.sqrt(var_bseret)
nse_meanret=(sum(nse_ret)/59)*12
bse_meanret=(sum(bse_ret)/59)*12


def main(type): 
  s=['bse','nse','non_nse']
  t=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if type==0:
    filepath=r"bsedata1.csv"
  if type==1:
    filepath=r"nsedata1.csv"
  if type==2:
    filepath=r"nse_non_index_data1.csv"
  print('\nAnalysis of',s[type],'stocks:\n')
  defcon1 = pd.read_csv(filepath)
  stocks=[]
  m=0
  for col in defcon1.columns:
    if m>0:
      stocks.append(col)
    m+=1
  n=len(stocks)
  returnn,betaa=[],[]
  for i in range(n):
    print("Name of the stock :",stocks[i])
    temp,tempret=[],[]
    for j in range(60):
      temp.append( defcon1.loc[j,stocks[i]])
      if j>0:
        tempret.append( (temp[j]-temp[j-1])/temp[j-1] )
    
    if type==1 or type==2:
       aa=sum(nse_ret)/len(nse_ret)
    else :
       aa=sum(bse_ret)/len(bse_ret)
    bb,c=sum(tempret)/len(tempret),0
    if type==1 or type==2:
      for k in range(len(tempret)):
         c+=(tempret[k]-bb)*(nse_ret[k]-aa)
      c=c/(len(tempret))
      c=c/var_nseret
      beta=c
      betaa.append(beta)
      st_ret=bb*12
      returnn.append(st_ret)
      res=st_ret-(0.1+(nse_meanret-0.1)*beta)
      if res>0:
        print("The above mentioned stock is undervalued")
      elif res<0:
        print("The above mentioned stock is overvalued")
      elif res==0:
        print("The above mentioned stock lies on the Sml")
    else :
      for k in range(len(tempret)):
         c+=(tempret[k]-bb)*(bse_ret[k]-aa)
      c=c/(len(tempret))
      c=c/var_bseret
      beta=c
      st_ret=bb*12
      betaa.append(beta)
      returnn.append(st_ret)
      res=st_ret-(0.1+(bse_meanret-0.1)*beta)
      if res>0:
        print("The above mentioned stock is undervalued")
        print(" ")
      elif res<0:
        print("The above mentioned stock is overvalued")
        print(" ")
      elif res==0:
        print("The above mentioned stock lies on the sml")
        print(" ")
  print('\n\n')
  print('Security Market Line')
  bet,rett=[],[]
  if type==1 or type==2:
    mark=nse_meanret
    mag='NSE'
  else:
    mark=bse_meanret
    mag='BSE'
  for hg in range(80):
    bet.append(0.05*hg)
    rett.append(0.1+bet[hg]*(mark-0.1))   
  plt.plot(bet,rett)
  plt.scatter(betaa,returnn) 
  for hj in range(10):
    plt.annotate(stocks[hj],(betaa[hj],returnn[hj]))
  
  plt.xlabel('Beta')
  plt.ylabel('Return')
  plt.show()
  print("The chosen Market portfolio is :",mag)
  print('\n\n',"***********",'\n\n')

  
main(0)
main(1)
main(2)


# In[ ]:




