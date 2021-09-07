#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.stats import norm

def positive(S,K,del_t,r,sigma):
    val = math.log(S/K) + (r+(sigma*sigma/2))*(del_t)
    return val/(sigma*math.sqrt(del_t))


def negative(S,K,del_t,r,sigma):
    val = math.log(S/K) + (r-(sigma*sigma/2))*(del_t)
    return val/(sigma*math.sqrt(del_t))

def BSM_call_option(S,K,T,t,r,sigma):
    if(t==T):
        return np.maximum(S-K,0)
    term1 = S*norm.cdf(positive(S,K,T-t,r,sigma)) 
    term2 = K*math.exp(-r*(T-t))*norm.cdf(negative(S,K,T-t,r,sigma))
    return term1-term2
  
def BSM_put_option(S,K,T,t,r,sigma):
    if(t==T):
        return np.maximum(K-S,0)
    return K*math.exp(-r*(T-t))-S+BSM_call_option(S,K,T,t,r,sigma)

T = 1;
K = 1;
r = 0.05;
sigma = 0.6
t = 0.5
x = 1

BSM_call = [];
BSM_put = []

u = np.linspace(0.5,1.5,100)
for i in u:
    BSM_put.append(BSM_put_option(x,i,T,t,r,sigma))
    BSM_call.append(BSM_call_option(x,i,T,t,r,sigma))

print ('Strike Price(K)       Call Option Price(C(t,x))     Put Option Price(P(t,x))')
for i in range(0,len(u),10):
    print(round(u[i],6),"                  ",round(BSM_call[i],6),"                       ",round(BSM_put[i],6))
plt.plot(u,BSM_put,label='P(t,x)',c='b')    
plt.plot(u,BSM_call,label='C(t,x)',c='g')

plt.title('C(t,x) and P(t,x) varying K (Strike Price)')
plt.xlabel('K (Strike Price)')
plt.ylabel('Option Price')
plt.legend()
plt.show()



BSM_call = [];
BSM_put = []
v = np.linspace(0.01,1,100)

for i in v:
    BSM_put.append(BSM_put_option(x,K,T,t,r,i))
    BSM_call.append(BSM_call_option(x,K,T,t,r,i))
print ('Sigma       Call Option Price(C(t,x))     Put Option Price(P(t,x))')
for i in range(0,len(v),10):
    print(round(v[i],6),"              ",round(BSM_call[i],6),"                       ",round(BSM_put[i],6))

plt.plot(v,BSM_put,label='P(t,x)',c='b')    
plt.plot(v,BSM_call,label='C(t,x)',c='g')

plt.title('C(t,x) and P(t,x) varying sigma')
plt.xlabel('Sigma')
plt.ylabel('Option Price')
plt.legend()
plt.show()


BSM_call = [];
BSM_put = []
w = np.linspace(0,1,100)

for i in w:
    BSM_put.append(BSM_put_option(x,K,T,t,i,sigma))
    BSM_call.append(BSM_call_option(x,K,T,t,i,sigma))
print ('Rate      Call Option Price(C(t,x))      Put Option Price(P(t,x))')
for i in range(0,len(w),10):
    print(round(w[i],6),"              ",round(BSM_call[i],6),"                       ",round(BSM_put[i],6))
plt.plot(w,BSM_put,label='P(t,x)',c='b')    
plt.plot(w,BSM_call,label='C(t,x)',c='g')

plt.title('C(t,x) and P(t,x) varying rate (r)')
plt.xlabel('Rate (r)')
plt.ylabel('Option Price')
plt.legend()
plt.show()


BSM_call = [];BSM_put = []
y = np.linspace(0.51,5,100)

for i in y:
    BSM_put.append(BSM_put_option(x,K,i,t,r,sigma))
    BSM_call.append(BSM_call_option(x,K,i,t,r,sigma))
print ('T    Call Option Price(C(t,x))   Put Option Price(P(t,x))')
for i in range(0,len(y),10):
    print(round(y[i],6),"              ",round(BSM_call[i],6),"                       ",round(BSM_put[i],6))
plt.plot(y,BSM_put,label='P(t,x)',c='b')    
plt.plot(y,BSM_call,label='C(t,x)',c='g')

plt.title('C(t,x) and P(t,x) varying Final Time (T)')
plt.xlabel('Final Time (T)')
plt.ylabel('Option Price')
plt.legend()
plt.show()



BSM_call={}
BSM_put={}
m=np.linspace(0.5,1.5,20)
l=[]
n=np.linspace(0.01,1,20)
k=[]
BSM_call_t=[]
BSM_put_t=[]

for i in range(0,len(n)):
    for j in range(0,len(m)):
        BSM_call[i,j] = BSM_call_option(x,m[j],T,t,r,n[i])
        BSM_put[i,j] = BSM_put_option(x,m[j],T,t,r,n[i])
        l.append(m[j]);k.append(n[i]);
        BSM_call_t.append(BSM_call[i,j]);BSM_put_t.append(BSM_put[i,j])
print ('Strike Price(K)     Sigma    Call Option Price(C(t,x))   Put Option Price(P(t,x))')
for i in range(0,20,2):
    print(round(m[i],6),"         ",round(n[i],6),"              ",round(BSM_call[i,i],6),"                       ",round(BSM_put[i,i],6))
axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_call_t, (len(n), len(m)))
axes.plot_surface(X, Y, Z,cmap ='plasma', edgecolor ='green')

axes.set_title('3D plot of C(t,x) varying K and sigma' )
axes.set_xlabel('K (Strike Price)') 
axes.set_ylabel('sigma')
axes.set_zlabel('C(t,x)')
axes.view_init(40, 60)
plt.show()

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_put_t, (len(n), len(m)))
axes.view_init(40, 210)
axes.plot_surface(X, Y, Z, cmap ='viridis', edgecolor='pink')
axes.set_title('3D plot of P(t,x) varying K and sigma' )
axes.set_xlabel('K (Strike Price)') 
axes.set_ylabel('sigma')
axes.set_zlabel('P(t,x)')
plt.show()



BSM_call={}
BSM_put={}
m=np.linspace(0.5,1.5,20)
l=[]
n=np.linspace(0,1,20)
k=[]
BSM_call_t=[]
BSM_put_t=[]

for i in range(0,len(n)):
    for j in range(0,len(m)):
        BSM_call[i,j] = BSM_call_option(x,m[j],T,t,n[i],sigma)
        BSM_put[i,j] = BSM_put_option(x,m[j],T,t,n[i],sigma)
        l.append(m[j]);k.append(n[i]);
        BSM_call_t.append(BSM_call[i,j]);BSM_put_t.append(BSM_put[i,j])
print ('Strike Price(K)       Rate        Call Option Price(C(t,x))     Put Option Price(P(t,x))')
for i in range(0,20,2):
    print(round(m[i],6),"         ",round(n[i],6),"              ",round(BSM_call[i,i],6),"                       ",round(BSM_put[i,i],6))

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_call_t, (len(n), len(m)))
axes.plot_surface(X, Y, Z,cmap ='plasma', edgecolor ='green')

axes.set_title('3D plot of C(t,x) varying K and r' )
axes.set_xlabel('K (Strike Price)') 
axes.set_ylabel('r (rate)')
axes.set_zlabel('C(t,x)')
axes.view_init(40, 60)
plt.show()

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_put_t, (len(n), len(m)))
axes.view_init(40, 60)
axes.plot_surface(X, Y, Z, cmap ='viridis', edgecolor='pink')
axes.set_title('3D plot of P(t,x) varying K and r' )
axes.set_xlabel('K (Strike Price)') 
axes.set_ylabel('r (rate)')
axes.set_zlabel('P(t,x)')
plt.show()



BSM_call={}
BSM_put={}
m=np.linspace(0.5,1.5,20)
l=[]
n=np.linspace(0.5,5,20)
k=[]
BSM_call_t=[]
BSM_put_t=[]

for i in range(0,len(n)):
    for j in range(0,len(m)):
        BSM_call[i,j] = BSM_call_option(x,m[j],n[i],t,r,sigma)
        BSM_put[i,j] = BSM_put_option(x,m[j],n[i],t,r,sigma)
        l.append(m[j]);k.append(n[i]);
        BSM_call_t.append(BSM_call[i,j]);BSM_put_t.append(BSM_put[i,j])
print ('Strike Price(K)       T      Call Option Price(C(t,x))   Put Option Price(P(t,x))')
for i in range(0,20,2):
    print(round(m[i],6),"         ",round(n[i],6),"              ",round(BSM_call[i,i],6),"                       ",round(BSM_put[i,i],6))

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_call_t, (len(n), len(m)))
axes.plot_surface(X, Y, Z,cmap ='plasma', edgecolor ='green')

axes.set_title('3D plot of C(t,x) varying K and T' )
axes.set_xlabel('K (Strike Price)') 
axes.set_ylabel('T (Final Time)')
axes.set_zlabel('C(t,x)')
axes.view_init(40, 60)
plt.show()

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_put_t, (len(n), len(m)))
axes.view_init(40, 210)
axes.plot_surface(X, Y, Z, cmap ='viridis', edgecolor='pink')
axes.set_title('3D plot of P(t,x) varying K and T' )
axes.set_xlabel('K (Strike Price)') 
axes.set_ylabel('T (Final Time)')
axes.set_zlabel('P(t,x)')
plt.show()



BSM_call={}
BSM_put={}
m=np.linspace(0,1,100)
l=[]
n=np.linspace(0.5,5,20)
k=[]
BSM_call_t=[]
BSM_put_t=[]

for i in range(0,len(n)):
    for j in range(0,len(m)):
        BSM_call[i,j] = BSM_call_option(x,K,n[i],t,m[j],sigma)
        BSM_put[i,j] = BSM_put_option(x,K,n[i],t,m[j],sigma)
        l.append(m[j]);k.append(n[i]);
        BSM_call_t.append(BSM_call[i,j]);BSM_put_t.append(BSM_put[i,j])
print ('Rate         T       Call Option Price(C(t,x))   Put Option Price(P(t,x))')
for i in range(0,20,2):
    print(round(m[5*i],6),"         ",round(n[i],6),"              ",round(BSM_call[i,5*i],6),"                       ",round(BSM_put[i,5*i],6))
axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_call_t, (len(n), len(m)))
axes.plot_surface(X, Y, Z,cmap ='plasma', edgecolor ='green')

axes.set_title('3D plot of C(t,x) varying r (rate) and T(Final Time)' )
axes.set_xlabel('r (rate)') 
axes.set_ylabel('T (Final Time)')
axes.set_zlabel('C(t,x)')

plt.show()

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_put_t, (len(n), len(m)))

axes.plot_surface(X, Y, Z, cmap ='viridis', edgecolor='yellow')
axes.set_title('3D plot of P(t,x) varying with r(rate) and T(Final Time)' )
axes.set_xlabel('r (rate)') 
axes.set_ylabel('T (Final Time)')
axes.set_zlabel('P(t,x)')
plt.show()



BSM_call={}
BSM_put={}
m=np.linspace(0,1,100)
l=[]
n=np.linspace(0.01,1,20)
k=[]
BSM_call_t=[]
BSM_put_t=[]

for i in range(0,len(n)):
    for j in range(0,len(m)):
        BSM_call[i,j] = BSM_call_option(x,K,T,t,m[j],n[i])
        BSM_put[i,j] = BSM_put_option(x,K,T,t,m[j],n[i])
        l.append(m[j]);k.append(n[i]);
        BSM_call_t.append(BSM_call[i,j]);BSM_put_t.append(BSM_put[i,j])
print ('Rate             Sigma         Call Option Price(C(t,x))     Put Option Price(P(t,x))')
for i in range(0,20,2): 
    print(round(m[5*i],6),"         ",round(n[i],6),"              ",round(BSM_call[i,5*i],6),"                       ",round(BSM_put[i,5*i],6))
axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_call_t, (len(n), len(m)))
axes.plot_surface(X, Y, Z,cmap ='viridis', edgecolor ='green')

axes.set_title('3D plot of C(t,x) varying with r and sigma' )
axes.set_xlabel('r (rate)') 
axes.set_ylabel('sigma')
axes.set_zlabel('C(t,x)')
axes.view_init(40, 210)
plt.show()

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_put_t, (len(n), len(m)))
axes.view_init(40, 30)
axes.plot_surface(X, Y, Z, cmap ='inferno', edgecolor='pink')
axes.set_title('3D plot of P(t,x) varying with r and sigma' )
axes.set_xlabel('r (rate)') 
axes.set_ylabel('sigma')
axes.set_zlabel('P(t,x)')
plt.show()


BSM_call={}
BSM_put={}
m=np.linspace(0.5,5,20)
l=[]
n=np.linspace(0.01,1,20)
k=[]
BSM_call_t=[]
BSM_put_t=[]

for i in range(0,len(n)):
    for j in range(0,len(m)):
        BSM_call[i,j] = BSM_call_option(x,K,T,t,m[j],n[i])
        BSM_put[i,j] = BSM_put_option(x,K,T,t,m[j],n[i])
        l.append(m[j]);k.append(n[i]);
        BSM_call_t.append(BSM_call[i,j]);BSM_put_t.append(BSM_put[i,j])
print ('T      Sigma    Call Option Price(C(t,x))   Put Option Price(P(t,x))')
for i in range(0,20,2):
    print(round(m[i],6),"         ",round(n[i],6),"              ",round(BSM_call[i,i],6),"                       ",round(BSM_put[i,i],6))
axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_call_t, (len(n), len(m)))
axes.plot_surface(X, Y, Z,cmap ='viridis', edgecolor ='green')

axes.set_title('3D plot of C(t,x) varying with T  and sigma' )
axes.set_xlabel('T (Final Time)') 
axes.set_ylabel('sigma')
axes.set_zlabel('C(t,x)')
axes.view_init(40, 210)
plt.show()

axes = plt.axes(projection ='3d') 
X = np.reshape(l, (len(n), len(m)))
Y = np.reshape(k, (len(n), len(m)))
Z = np.reshape(BSM_put_t, (len(n), len(m)))
axes.view_init(40, 30)
axes.plot_surface(X, Y, Z, cmap ='inferno', edgecolor='pink')
axes.set_title('3D plot of P(t,x) varying with T  and sigma' )
axes.set_xlabel('T (Final time)') 
axes.set_ylabel('sigma')
axes.set_zlabel('P(t,x)')
plt.show()


# In[ ]:




