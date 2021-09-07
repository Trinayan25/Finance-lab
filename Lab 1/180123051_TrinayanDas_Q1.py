#!/usr/bin/env python
# coding: utf-8

# In[4]:




import math


T=5
sigma=0.3
r=0.05
K=105
S0=100

def auxilary_put(Pri):
  if (Pri<=K):
    return (K-Pri)
  elif (Pri>K):
    return 0


def auxilary_call(Pri):
  if (Pri>=K):
    return (-K+Pri)
  elif (Pri<K):
    return 0

def fact(i):
  prod=1
  while i>1:
    prod=prod*i
    i=i-1
  return prod
def Price(m,a,b,f):
  Price_c=S0
  Price_w=pow(a,m)*pow(b,f-m)
  Price_c=Price_c*Price_w
  return Price_c

def combination(i,r):
  k_1=fact(i)
  k_2=fact(i-r)*fact(r)
  k_1=k_1/k_2
  return k_1

 

M=[1,5,10,20,50,100,200,400]
call=[]
put=[]

def main(f):
  delta=T/f
  delta_1=math.sqrt(delta)
  a=math.exp(sigma*delta_1+(r-(sigma*sigma)/2)*delta)
  b=math.exp(-sigma*delta_1+(r-(sigma*sigma)/2)*delta)
  p=((math.exp(r*delta))-b)/(a-b)
  q=1-p
  base=1/(math.exp(r*T))
  sum_1=0
  sum_2=0
  for m in range(f+1):
    price_o=Price(m,a,b,f)
    price=auxilary_call(price_o)
    price=price*combination(f,m)
    price=price*pow(p,m)*pow(q,f-m)
    sum_1=sum_1+price
    price_o=Price(m,a,b,f)
    price=auxilary_put(price_o)
    price=price*combination(f,m)
    price=price*pow(p,m)*pow(q,f-m)
    sum_2=sum_2+price
  sum_1=sum_1*base
  sum_2=sum_2*base
  call.append(sum_1)
  put.append(sum_2)

    
for k in M:
  main(k)



for m in range(len(M)):
  print("For the value of M =",M[m])
  delta_8=T/M[m]
  delta_11=math.sqrt(delta_8)
  alpha=math.exp(sigma*delta_11+(r-(sigma*sigma)/2)*delta_8)
  beta=math.exp(-sigma*delta_11+(r-(sigma*sigma)/2)*delta_8)
  if (alpha>math.exp(r*delta_8)) and (math.exp(r*delta_8)>beta):
     print(' The call price at time 0 =',call[m],', The put price at time 0 =',put[m])
     print(' ')
     print(' ')
  else :
    print('The noo arbitrage condition violated for M =',M[m])



# In[ ]:




