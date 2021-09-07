#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Question no 3
import math
import numpy as np
import matplotlib.pyplot as plt

u = [1,1,1,1,1,1,1,1,1,1]
v=np.transpose(u)
m = [0.4067,0.975,0.7804,-0.4696,0.2811,0.1796,0.06869,0.4289,0.0764,0.1094]
n=np.transpose(m)
c = [[36287.3174,4014.457,84026.129,137.9479,6266.541,1930.082,2343.791,3338.262,618.222,3138.737],
     [4014.4575,550.675,12428.681,-428.197,808.234,257.617,277.858,408.038,78.854,308.338],
     [84026.129,12428.681,330344.305,-21151.436,19084.886,6151.104,5133.395,8184.932,1677.820,3762.625],
     [137.947,-428.197,-21151.436,5159.857,-472.838,-307.813,354.504,12.551,0.6983,826.690],
     [6266.541,808.234,19084.886,-472.838,1379.758,383.586,471.608,604.541,117.149,442.445],
     [1930.082,257.617,6151.104,-307.813,383.586,151.873,126.875,206.289,39.135,133.422],
     [2343.791,277.858,5133.395,354.5049,471.608,126.875,290.328,260.016,48.510,242.067],
     [338.262,408.038,8184.932,12.551,604.541,206.289,260.016,373.680,70.461,329.513],
     [618.222,78.854,1677.820,0.698,117.149,39.135,48.510,70.461,16.475,66.804],
     [3138.737,308.3385,3762.625,826.690,442.445,133.4228,242.0678,329.513,66.8047,496.9038]]
k=np.linalg.inv(c)

return_1=[]
risk_1=[]



wmin= np.dot(u,k)
global_return=0
zeta=np.dot(np.dot(u,k),v)
for i in range(10):
  wmin[i]=wmin[i]/zeta
  global_return+=wmin[i]*m[i]

for i in range(101):
    return_1.append(i*0.004+global_return)
    r=[return_1[i],1]
    M=[[ np.dot(np.dot(m,k),n), np.dot(np.dot(u,k),n)], [np.dot(np.dot(m,k),v), np.dot(np.dot(u,k),v)]]
    lam=np.dot(np.linalg.inv(M),r)
    a=lam[0]
    b=lam[1]
    q,y,w=[],[],[]
    for j in range(10):
      q.append(m[j]*a)
      y.append(u[j]*b)
    w_1=np.dot(q,k)
    w_2=np.dot(y,k)
    for j in range(10):
      w.append(w_1[j]+w_2[j])
    d=np.transpose(w)
    var=np.dot(np.dot(w,c),d)
    sigma=math.sqrt(var)
    risk_1.append(sigma)

print("Efficient frontier")
print(" ")
plt.plot(risk_1,return_1,'r')
plt.xlabel('Portfolio Risk ')
plt.ylabel('Portfolio Return ')
plt.show()

risk_free=0.05
imp=[]
for i in range(10):
  imp.append(m[i]-risk_free)
mpw_phase1=np.dot(np.dot(imp,k),v)
market_portfolio=np.dot(imp,k)
for i in range(10):
  market_portfolio[i]=market_portfolio[i]/mpw_phase1
ret_market=0
for i in range(10):
  ret_market+=m[i]*market_portfolio[i]

msig=math.sqrt(np.dot(np.dot(market_portfolio,c),np.transpose(market_portfolio)))

print(" ")
print("The market portfolio has return =",ret_market*100,'%')
print("and weights")
for s in range(10):
  print('w',s,' =',market_portfolio[s])
print("and risk =", math.sqrt(np.dot(np.dot(market_portfolio,c),np.transpose(market_portfolio)))*100,'%')
f,s=[],[]
print(" ")
print(" ")
for i in range(1000):
  f.append(0.003*i)
  s.append(risk_free+((ret_market-risk_free)*f[i])/msig)
kj=(ret_market-risk_free)/msig

print("The slope & intercept of the Cml are ",kj,'and',risk_free)
print(" ")
print("Capital Market line  to Efficient Frontier")
print(" ")
plt.plot(risk_1,return_1,'r',label='Efficient Frontier')
plt.plot(f,s,'b',label='Capital market line')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return ')
plt.legend(loc='lower right')
plt.show()

print('  ')
print('The slope & intercept of the security market line:', ret_market-risk_free,'and',risk_free)
print(' ')
alicia,keys=[],[]
for i in range(100):
  alicia.append(i*0.06)
  keys.append((ret_market-risk_free)*alicia[i]+risk_free)
print("Security Market Line")
print(' ')
plt.plot(alicia,keys,'r')
plt.xlabel('Beta')
plt.ylabel('Return')
plt.show()


# In[ ]:




