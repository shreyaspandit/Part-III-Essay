import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import beta, betabinom

def f(x, a, b):
    return a/(1+(x-b)**2)

def mu(x):
    return f(x, 1, 0) + f(x, 2, 3)

def generate_data(n):
    x = np.random.uniform(-5, 5, n)
    y = mu(x) + 0.5*np.random.randn(n)
    return np.column_stack((x, y)) # return matrix with X, Y in first/second column

def absolute_residual_score(x, y, model):
    return np.abs(y - model.predict(x).flatten())

def check_coverage(alpha, n, y, y_lower, y_upper, q_cal):
    if q_cal >= n+1:
        return 1
    else:
        return np.mean((y <= y_upper) & (y >= y_lower))

R = 7000 # number of calibration/test sets we average over
n_train, n_cal, n_test = 1000, 1500, 1500
n = n_cal + n_test

D_train = generate_data(n_train)
X_train, Y_train = D_train[:, 0], D_train[:, 1]

alpha = 0.1
q_cal = int(np.ceil((1-alpha)*(n_cal + 1)))

# Linear regression fit ------------------------------------------------------

model_LR = LinearRegression()
model_LR.fit(X_train.reshape(-1,1), Y_train.ravel())

cov_LR = np.zeros((R, ))
for r in range(R):
    D = generate_data(n)
    D_cal = D[:n_cal,:]
    D_test = D[n_cal:(n_cal + n_test),:]
    X_cal, Y_cal = D_cal[:, 0], D_cal[:, 1]
    X_test_LR, Y_test_LR = D_test[:, 0], D_test[:, 1]

    scores_LR = absolute_residual_score(X_cal.reshape(-1, 1), Y_cal, model_LR)
    Y_pred_LR = model_LR.predict(X_test_LR.reshape(-1, 1)).flatten()
    q_LR = np.sort(scores_LR.flatten())[q_cal - 1]
    y_upper_LR = Y_pred_LR + q_LR
    y_lower_LR = Y_pred_LR - q_LR

    cov_LR[r] = check_coverage(alpha, n_cal, Y_test_LR, y_lower_LR, y_upper_LR, q_cal)

print(str(np.mean(cov_LR)))

k = np.ceil((1-alpha)*(n_cal+1))
x = np.array(range(int(n_test*(0.85)),int(n_test*0.95)))
beta_bin_dist = betabinom(n_test, k, n_cal+1-k)

plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r"$x/n_{\mathrm{test}}$")
ax.set_ylabel(r"$p(x; n_{\mathrm{test}}, n, k_{\alpha})$")
ax.plot(x/n_test, beta_bin_dist.pmf(x))
ax.hist(cov_LR, bins=x/n_test, weights=(1/R)*np.ones_like(cov_LR), label="Empirical coverage")
ax.legend()

plt.tight_layout()
fig.savefig("2_3_training_beta_dist.png")
plt.show()