import numpy as np
import matplotlib.pyplot as plt
import time

######################load the data###########################
data = np.loadtxt('data.txt')
X = np.array(data[:, 0:8])
y = np.array(data[:, 8]).T

# The number of features.
m = X.shape[0]
n = X.shape[1]
print("number of features:",n)
w = np.zeros(n)
C = 10
ep = 0.03
eta = 0.000000001
b = 0
eta2 = 0.00000001
ep2 = 0.4
eta3 = 0.00000001
ep3 = 0.5

#############################Loss function###################################################
def loss(X, y, w,b, C):
    m = X.shape[0]
    cost = (1/2) * np.dot(w.T, w) + (C * np.sum(np.maximum(np.zeros(m), 1 - y * (X@w + b))))
    return cost.item()

################################# Batch Gradient Descent algorithm###########################
def batchgradientDescent(X, y, C, ep, eta):
    m = X.shape[0]
    n = X.shape[1]
    w = np.zeros(n)
    new_w = np.zeros(n)
    b = 0
    new_b = 0
    costHistory = []
    costHistory.append(loss(X, y, w, b, C))
    k = 0
    while True:
        parGrad = np.zeros(n)
        flag = y * (X @ w + b) < 1
        flag = flag.astype(int)
        for j in range(n):
             parGrad[j] = np.sum(flag * (-y * X[:,j]))
             grad_w = w[j] + (C * parGrad[j])
             new_w[j] = w[j] - eta * grad_w
        parGrad_b = np.sum(flag * (-y))
        grad_b = C * parGrad_b
        new_b = b - eta * grad_b
        w = new_w
        b = new_b
        costHistory.append(loss(X, y, w,b, C))
        k = k + 1
        # convergence critrion
        if abs(costHistory[k - 1]-costHistory[k]) * 100 / costHistory[k - 1] < ep:
            break
    return costHistory, w, b, k

#

###############################stochastic gradient descent Algorithm###############################
def stochasticgradientDescent(X, y, C, ep2, eta2):
    m = X.shape[0]
    n = X.shape[1]
    w = np.zeros(n)
    new_w = np.zeros(n)
    b = 0
    new_b = 0
    costHistory = []
    previous_cost = loss(X, y, w, b, C)
    costHistory.append(previous_cost)
    k = 0
    while True:
        parGrad = np.zeros(n)
        for i in range(0, m):
            flag = y[i] * (X[i] @ w + b) < 1
            flag = flag.astype(int)
            for j in range(n):
                 parGrad[j] = (flag * (-y[i] * X[i][j]))
                 grad_w = w[j] + C * parGrad[j]
                 new_w[j] = w[j] - eta2 * grad_w
            parGrad_b = (flag * (-y[i]))
            grad_b = C * parGrad_b
            new_b = b - eta2 * grad_b
            w = new_w
            b = new_b
            costHistory.append(loss(X, y, w, b, C))
            k = k+1
        # convergence critrion
        cost = loss(X, y, w, b, C)
        if abs(previous_cost - cost) * 100 / previous_cost < ep2:
            break
        previous_cost = cost

    return costHistory, w, b, k




################################# mini batch gradient descent algorithm#################################
def minibatchgradientDescent(X, y, C, ep3, eta3):
    m = X.shape[0]
    n = X.shape[1]
    w = np.zeros(n)
    new_w = np.zeros(n)
    b = 0
    new_b = 0
    batch_size = 4
    costHistory = []
    previous_cost = loss(X, y, w, b, C)
    costHistory.append(previous_cost)
    i = 0
    k = 0
    while True:
        parGrad = np.zeros(n)
        for i in range(0, m, batch_size):
            X_batch = X[i:(i+batch_size)]
            y_batch = y[i:(i+batch_size)]
            flag = y_batch * (X_batch @ w + b) < 1
            flag = flag.astype(int)
            for j in range(n):
                 parGrad[j] = np.sum(flag * (-y_batch * X_batch[:,j]))
                 grad_w = w[j] + (C * parGrad[j])
                 new_w[j] = w[j] - eta3 * grad_w
            parGrad_b = np.sum(flag * (-y_batch))
            grad_b = C * parGrad_b
            new_b = b - eta3 * grad_b
            w = new_w
            b = new_b
            costHistory.append(loss(X, y, w, b, C))
            k = k + 1
        # convergence critrion
        cost = loss(X, y, w, b, C)
        if abs(previous_cost - cost) * 100 / previous_cost < ep3:
            break
        previous_cost = cost

    return costHistory, w, b, k

########################## calling the functions ################################

start = time.time()
cost, w1, b1, k1 = batchgradientDescent(X, y, C, ep, eta)
end = time.time()
print("The time taken by BGD to converge is:", (end - start))
print("The cost values of BGD",cost)



start2 = time.time()
cost2, w2, b2, k2 = stochasticgradientDescent(X, y, C, ep2, eta2)
end2 = time.time()
print("The time taken by SGD to converge is:", end2 - start2)
print("The cost values of SGD",cost2)


start3 = time.time()
cost3, w3, b3, k3 = minibatchgradientDescent(X, y, C, ep3, eta3)
end3 = time.time()
print("The time taken by MBGD to converge is:", end3 - start3)
print("The cost values of MBGD", cost3)


################### plotting the 3 graphs in graph ###############################
plt.plot(cost)
plt.plot(cost2)
plt.plot(cost3)
plt.legend(['BGD', 'SGD', 'MGBD'])
plt.show()
######## using subplots plot on same graph##############################
plt.subplot(1, 3, 1)
plt.plot(cost)
plt.title("BGD")
plt.subplot(1,3,2)
plt.plot(cost2)
plt.title("SGD")
plt.subplot(1, 3, 3)
plt.plot(cost3)
plt.title("MBGD")
plt.show()
