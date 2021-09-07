#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Question No 1

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

mu = np.array([0.1, 0.2, 0.15])
covariance_mat = np.array([[0.005, -0.010, 0.004], 
                [-0.010, 0.040, -0.002], 
                [0.004, -0.002, 0.023]])
risk_free = 0.1
dim = len(mu)
u = np.ones((1, dim))

def get_ret(w):
    return np.dot(w, mu)

def get_risk(w):
    return (np.matmul(np.matmul(w, covariance_mat), np.transpose(w)))**0.5

def model(Y):
    r = 0
    x = 0
    y = 0
    weights = []
    W = []
    X = []
    for i in range(len(Y)):
        m = Y[i]
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1},
            {'type': 'eq', 'fun': lambda w: get_ret(w)-m}
        )
        res = minimize(get_risk, np.array([0.2, 0.3, 0.5]), method='SLSQP', constraints=cons)
        X.append(res.fun)
        W.append(res.x)
        if (Y[i]-0.1)/X[i] > r:
            r = (Y[i]-0.1)/X[i]
            x = X[i]
            y = Y[i]
            weights = W[i]
    return W, X, x, y, weights

Y = np.linspace(0.005, 0.3, 1000)
W, X, market_risk, market_ret, market_w = model(Y)

print("Portfolio without riskfree assets at 15% risk")
print("Index\tWeight\t\t\t\t\tReturn\t\t\tRisk")
tol = 0.0003
for i in range(len(X)):
    if abs(X[i]-0.15) < tol:
        print(i, W[i], Y[i], X[i])
print()
        
print("Portfolio without riskfree assets at 18% return")
print("Index\tWeight\t\t\t\tReturn\t\t\tRisk")
tol = 0.00015
for i in range(len(X)):
    if abs(Y[i]-0.18) < tol:
        print(i, W[i], Y[i], X[i])
        break
print()
        
plt.plot(X, Y, color="yellow")
plt.axvline(x=0.15, color="green")
plt.axhline(y=0.18, color="purple")
plt.text(0.15,0,'x = 0.15')
plt.text(0,0.18,'y = 0.18')

indx = np.linspace(0, len(W)-1, 10)
indx = [int(i) for i in indx]

print("Index\tWeight\t\t\t\t\t\tReturn\t\t\tRisk")
for i in indx:
    print(str(i)+"\t"+str(W[i])+"\t"+"{0:.17f}".format(Y[i])+"\t"+"{0:.15f}".format(X[i]))
print()

risk_1 = 0.10
f_1 = risk_1/market_risk
w_1 = np.append(f_1*market_w, (1-f_1))
print("Portfolio with risky and riskfree assets at "+str(100*risk_1)+"% risk = ", end='')
print(w_1)

risk_2 = 0.25
f_2 = risk_2/market_risk
w_2 = np.append(f_2*market_w, (1-f_2))
print("Portfolio with risky and riskfree assets at "+str(100*risk_2)+"% risk = ", end='')
print(w_2)

plt.scatter(market_risk, market_ret, color='red')
plt.scatter(0, risk_free, color='red')

y_max = 0.3
x_max = market_risk+market_risk*(y_max-market_ret)/(market_ret-risk_free)

plt.plot(np.array([0, x_max]), np.array([risk_free, y_max]), color='red')
plt.annotate("  Market Portfolio("+str(round(market_risk, 3))+","+str(round(market_ret, 3))+")", (market_risk, market_ret))
plt.annotate("  Zero Risk Portfolio("+str(0)+","+str(risk_free)+")", (0, risk_free))
plt.title("Markowitz Efficient Frontier & CAPM Line")
plt.xlabel("Vol")
plt.ylabel("Return")
plt.show()


# In[ ]:




