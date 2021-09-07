#!/usr/bin/env python
# coding: utf-8

# In[15]:


import math
import matplotlib.pyplot as plt


T=5
sigma=0.3
r=0.05
K=105
S0=100
delta=T/20
delta_1=math.sqrt(delta)

def aux_put(Pri):
  if (Pri<=K):
    return (K-Pri)
  elif (Pri>K):
    return 0


def aux_call(Pri):
  if (Pri>=K):
    return (-K+Pri)
  elif (Pri<K):
    return 0

def fact(n):
  prod=1
  while n>1:
    prod=prod*n
    n=n-1
  return prod
def pric(i,a,b):
  pricc=S0
  pricw=pow(a,i)*pow(b,20-i)
  pricc=pricc*pricw
  return pricc

def combination(n,r):
  k_1=fact(n)
  k_2=fact(n-r)*fact(r)
  k_1=k_1/k_2
  return k_1

alpha=math.exp(sigma*delta_1+(r-(sigma*sigma)/2)*delta)
beta=math.exp(-sigma*delta_1+(r-(sigma*sigma)/2)*delta)
if (beta<math.exp(r*T/20)) and (alpha>math.exp(r*T/20)):
  print("The no arbitrage condition is verified")
  print(" ")

else :
  print("The calculations are invalid due to the violation of no arbitrage principle")

M=[0, 2, 4, 6, 12, 18]


def main(f):
  a=math.exp(sigma*delta_1+(r-(sigma*sigma)/2)*delta)
  b=math.exp(-sigma*delta_1+(r-(sigma*sigma)/2)*delta)
  p=((math.exp(r*delta))-b)/(a-b)
  q=1-p
  rem=((20-f)*T)/20
  base=1/math.exp(r*rem)
  print("The present time: ", T-rem)
  print("The remaining time: ", rem)
  print(" ")
  if f>=0:
    call=[]
    put=[]
    for s in range(f+1):
      up=s
      down=f-s
      s_1=0
      s_2=0
      for j in range(21-f):
        price_ok=pric(j+s,a,b)
        price1=aux_call(price_ok)
        price1=price1*combination(20-f,j)
        price1=price1*pow(p,j)*pow(q,20-f-j)
        s_1=s_1+price1
        price2=aux_put(price_ok)
        price2=price2*combination(20-f,j)
        price2=price2*pow(p,j)*pow(q,20-f-j)
        s_2=s_2+price2
      s_1=s_1*base
      s_2=s_2*base
      call.append(s_1)
      put.append(s_2)
    
    print("The call option prices:")
    print("The first entry indicates 0 ups and all downs, the second 1 up and the rest downs and so on")
    print(" ")
    print(call)
    print(" ")
    print("The put option prices:")
    print("The first entry indicates 0 ups and all downs, the second 1 up and the rest downs and so on")
    print(" ")
    print(put)
    print(" ")
    
for k in M:
  main(k)



# In[ ]:




