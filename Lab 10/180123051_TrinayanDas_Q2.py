#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Question No: 2

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (20,12)

def GBM(init, mu, sigma, T, dt, str):
  if str == 'Normal':
    no_values = round(T/dt)
    value = np.zeros([1,no_values])
    value[0][0] = init
    for i in range(1,no_values):
      c = (mu-0.5*sigma**2)
      value[0][i] = value[0][i-1]*np.exp(c*dt + sigma*np.sqrt(dt)*np.random.normal())
  elif str == 'var_reduce':
    no_values = round(T/dt)
    value_1 = np.zeros([1,no_values])
    value_2 = np.zeros([1,no_values])
    value_1[0][0] = init
    value_2[0][0] = init
    for i in range(1,no_values):
      c = (mu-0.5*sigma**2)
      value_1[0][i] = value_1[0][i-1]*np.exp(c*dt + sigma*np.sqrt(dt)*np.random.normal())   
      value_2[0][i] = value_2[0][i-1]*np.exp(c*dt + sigma*np.sqrt(dt)*np.random.normal())
    value = (value_1 + value_2)/2
  return value

init = 100
r = 0.05
mu = 0.1
sigma = 0.2
T = 0.5
dt = 0.001
path = 10
K = 105
value = np.zeros([round(T/dt),path])
str = 'var_reduce'
for i in range(0,path):
  value[:,i] = GBM(init, mu, sigma, T, dt, str)

print('Check for correctness :')
print('Mean :',np.mean(value[-1,:]))
print('Variance :',np.var(value[-1,:]))

x = np.linspace(0,T-dt,500)
plt.plot(x,value)
plt.xlabel('t')
plt.ylabel('Price')
plt.title('Real world')
plt.show()

for i in range(0,path):
  value[:,i] = GBM(init, r, sigma, T, dt, str)

print('Check for correctness :')
print('Mean :',np.mean(value[-1,:]))
print('Variance :',np.var(value[-1,:]))

x = np.linspace(0,T-dt,500)
plt.plot(x,value)
plt.xlabel('t')
plt.ylabel('Price')
plt.title('Risk-free world')
plt.show()

K = []
K.append(90)
K.append(105)
K.append(110)
path = 100

for i in range(0,3):
  call_price = 0
  put_price = 0
  for k in range(0,path):
    S = GBM(init, r, sigma, T, dt, str) 
    X = (sum(S[0])/(T/dt))-K[i]
    X1 = K[i]-(sum(S[0])/(T/dt))
    call_price = call_price + max(X,0)
    put_price = put_price + max(X1,0)
  call = (call_price/path)*np.exp(-r*T)
  put = (put_price/path)*np.exp(-r*T)
  print("The price of call option with strike price: %d is %f"%(K[i],call))
  print("The price of put option with strike price : %d is %f"%(K[i],put))

init = 100
r = 0.05
mu = 0.1
sigma = 0.2
T = 0.5
dt = 0.001
path = 100
K = 105

var = np.linspace(80,120,81)
path = 100
call = np.zeros([1,81])
put = np.zeros([1,81])

for i in range(0,81):
  call_price = 0
  put_price = 0
  for k in range(0,path):
    S = GBM(var[i], r, sigma, T, dt, str)
    f = (sum(S[0])/(T/dt))-K
    q = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(f,0)
    put_price = put_price + max(q,0)
  call[0][i] = (call_price/path)*np.exp(-r*T)
  put[0][i] = (put_price/path)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,call[0])
plt.xlabel('Initial Stock price')
plt.ylabel('Call-Price')

plt.subplot(2,1,2)
plt.plot(var,put[0])
plt.xlabel('Initial Stock price')
plt.ylabel('Put-Price')
plt.show()

path = 100
call = np.zeros([1,81])
put = np.zeros([1,81])

for i in range(0,81):
  call_price = 0
  put_price = 0
  for k in range(0,path):
    S = GBM(init, r, sigma, T, dt, str)
    f = (sum(S[0])/(T/dt))-var[i]
    q = var[i]-(sum(S[0])/(T/dt))
    call_price = call_price + max(f,0)
    put_price = put_price + max(q,0)
  call[0][i] = (call_price/path)*np.exp(-r*T)
  put[0][i] = (put_price/path)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,call[0])
plt.xlabel('Strike price')
plt.ylabel('Call Price')

plt.subplot(2,1,2)
plt.plot(var,put[0])
plt.xlabel('Strike price')
plt.ylabel('Put Price')
plt.show()

var = np.linspace(0.01,0.9,90)
path = 500
call = np.zeros([1,90])
put = np.zeros([1,90])

for i in range(0,90):
  call_price = 0
  put_price = 0
  for k in range(0,path):
    S = GBM(init, var[i], sigma, T, dt, str)
    f = (sum(S[0])/(T/dt))-K
    q = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(f,0)
    put_price = put_price + max(q,0)
  call[0][i] = (call_price/path)*np.exp(-r*T)
  put[0][i] = (put_price/path)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,call[0])
plt.xlabel('Risk-free price')
plt.ylabel('Call Price')

plt.subplot(2,1,2)
plt.plot(var,put[0])
plt.xlabel('Risk-free price')
plt.ylabel('Put Price')
plt.show()

var = np.linspace(0.01,0.9,90)
path = 500
call = np.zeros([1,90])
put = np.zeros([1,90])

for i in range(0,90):
  call_price = 0
  put_price = 0
  for k in range(0,path):
    S = GBM(init, r, var[i], T, dt, str)
    f = (sum(S[0])/(T/dt))-K
    q = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(f,0)
    put_price = put_price + max(q,0)
  call[0][i] = (call_price/path)*np.exp(-r*T)
  put[0][i] = (put_price/path)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,call[0])
plt.xlabel('Volatility')
plt.ylabel('Call-Price')

plt.subplot(2,1,2)
plt.plot(var,put[0])
plt.xlabel('Volatility')
plt.ylabel('Put-Price')
plt.show()
var = np.linspace(0.01,0.9,18)
path = 500
call = np.zeros([1,18])
put = np.zeros([1,18])

for i in range(0,18):
  call_price = 0
  put_price = 0
  for k in range(0,path):
    S = GBM(init, r, sigma, var[i], dt, str)
    f = (sum(S[0])/(T/dt))-K
    q = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(f,0)
    put_price = put_price + max(q,0)
  call[0][i] = (call_price/path)*np.exp(-r*T)
  put[0][i] = (put_price/path)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,call[0])
plt.xlabel('Maturity')
plt.ylabel('Call-Price')

plt.subplot(2,1,2)
plt.plot(var,put[0])
plt.xlabel('Maturity')
plt.ylabel('Put-Price')
plt.show()


# In[ ]:




