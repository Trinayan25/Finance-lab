#!/usr/bin/env python
# coding: utf-8

# In[12]:


import math
import matplotlib.pyplot as plt


T=5
sigma=0.3
r=0.05
K=105
S0=100

def auxilary_put(pri):
  if (pri<=K):
    return (K-pri)
  elif (pri>K):
    return 0


def auxilary_call(pri):
  if (pri>=K):
    return (-K+pri)
  elif (pri<K):
    return 0

def fact(n):
  prod=1
  while n>1:
    prod=prod*n
    n=n-1
  return prod
def Price(s,a,b,m):
  Price_c=S0
  Price_w=pow(a,s)*pow(b,m-s)
  Price_c=Price_c*Price_w
  return Price_c

def combination(n,r):
  k_1=fact(n)
  k_2=fact(n-r)*fact(r)
  k_1=k_1/k_2
  return k_1

print("M is ranging from 1 to 100")
print(" ")
print(' ')


M=[]
for s in range(101):
  M.append(s+1)
M2=[]
for s in range(21):
  M2.append(5*s+1)
call=[]
put=[]
call_1=[]
put_1=[]

def main(m,opt):
  delta=T/m
  delta1=math.sqrt(delta)
  a=math.exp(sigma*delta1+(r-(sigma*sigma)/2)*delta)
  b=math.exp(-sigma*delta1+(r-(sigma*sigma)/2)*delta)
  p=((math.exp(r*delta))-b)/(a-b)
  q=1-p
  if (a>math.exp(r*T/m)) and (b<math.exp(r*T/m)):
    base=1/(math.exp(r*T))
    sum_1=0
    sum2=0
    for s in range(m+1):
     price_o=Price(s,a,b,m)
     price=auxilary_call(price_o)
     price=price*combination(m,s)
     price=price*pow(p,s)*pow(q,m-s)
     sum_1=sum_1+price
     price_o=Price(s,a,b,m)
     price=auxilary_put(price_o)
     price=price*combination(m,s)
     price=price*pow(p,s)*pow(q,m-s)
     sum2=sum2+price
    sum_1=sum_1*base
    sum2=sum2*base
    if opt==1 :
     call.append(sum_1)
     put.append(sum2)
    if opt==2 :
     call_1.append(sum_1)
     put_1.append(sum2)
  
  else :
    print('The no arbitrage condition is violated; calculation terminated for M =',m)
 

    
for k in M:
  main(k,1)

plt.plot(M,call,label='Call')
plt.plot(M,put,label='Put')
plt.xlabel('M is increased by +1')
plt.ylabel('Price->')

plt.show()


print(' ')



    
for k in M2:
  main(k,2)

plt.plot(M2,call_1,label='Call')
plt.plot(M2,put_1,label='Put')
plt.xlabel('M is increased by +5')
plt.ylabel('Price->')

plt.show()


# In[ ]:




