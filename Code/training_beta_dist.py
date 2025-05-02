import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import beta, betabinom

random_seed = 25
np.random.seed(random_seed)

def f(x, a, b):
    return a/(1+(x-b)**2)

def mu(x):
    return f(x, 1, 0) + f(x, 2, 3)

def generate_data(n):
    x = np.random.uniform(-5, 5, n)
    y = mu(x) + 0.5*np.random.randn(n)
    return np.column_stack((x, y)) # return matrix with X, Y in first/second column

# Check coverage function

def get_prediction_set(Y_pred, S, alpha):
    n = len(S)
    q = np.quantile(S.flatten(), np.ceil((n+1)*(1-alpha))/n, method="higher")
    Y_lower = Y_pred - q
    Y_upper = Y_pred + q
    return np.stack((Y_lower, Y_upper), axis=1)

def check_coverage(R, alpha, n_cal, n_test, X_cal, Y_cal, X_test, Y_test, S, Y_pred):
    num_cal = int(n_cal/R)
    num_test = int(n_test/R)
    cov = np.zeros((R,))
    for j in range(R):
        x_cal, y_cal = X_cal[j*num_cal:(j+1)*num_cal], Y_cal[j*num_cal:(j+1)*num_cal]
        scores = S[j*num_cal:(j+1)*num_cal]

        x_test, y_test = X_test[j*num_test:(j+1)*num_test], Y_test[j*num_test:(j+1)*num_test]
        y_pred = Y_pred[j*num_test:(j+1)*num_test]

        pred_set = get_prediction_set(y_pred, scores, alpha)
        y_lower = pred_set[:, 0]
        y_upper = pred_set[:, 1]

        cov[j] = np.mean((y_test <= y_upper) & (y_test >= y_lower))

    return cov



# Generate proper training, calibration and test data

R = 7000

n_train = 1000
n_cal = 3000*R
n_test = 3000*R

n = n_train + n_cal + n_test
alpha = 0.1

D = generate_data(n)

D_train = D[:n_train,:]
D_cal = D[n_train:(n_train + n_cal),:]
D_test = D[(n_train + n_cal):(n_train + n_cal + n_test),:]

X_train = D_train[:, 0]
Y_train = D_train[:, 1]
X_cal = D_cal[:, 0]
Y_cal = D_cal[:, 1]
X_test = D_test[:, 0]
Y_test = D_test[:, 1]

def absolute_residual_score(x, y, model):
    return np.abs(y - model.predict(x).flatten()) # returns row vector

# Define the adjusted quantile
q_cal = int(np.ceil((1-alpha)*(n_cal + 1)))

# Fit and calibrate the linear regression model
model_LR = LinearRegression()
model_LR.fit(X_train.reshape(-1, 1), Y_train.ravel())

# Split conformal procedure:
scores_LR = absolute_residual_score(X_cal.reshape(-1, 1), Y_cal, model_LR)
q_LR = np.sort(scores_LR.flatten())[q_cal - 1]

Y_pred = model_LR.predict(X_test.reshape(-1,1)).flatten()

cov = check_coverage(R, alpha, n_cal, n_test, X_cal, Y_cal, X_test, Y_test, scores_LR, Y_pred)

N = int(n_cal/R)
N_val = int(n_test/R)
l = np.ceil((1-alpha)*(N+1))
a = l
b = N+1-l
# x = np.linspace(betabinom.ppf(0.001,N_val, a,b),beta.ppf(0.999,N_val, a,b), 1000)

x = np.array(range(int(N_val*(0.85)),int(N_val*0.95)))
rv = betabinom(N_val,a, b)
# Renormalize the pmf to account for the larger number of points.

plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r"$x/n_{\mathrm{test}}$")
ax.set_ylabel(r"$p(x; n_{\mathrm{test}}, n, k_{\alpha})$")
ax.plot(x/N_val, rv.pmf(x))
ax.hist(cov, bins=x/N_val, weights=(1/R)*np.ones_like(cov), label="Empirical coverage")
ax.legend()

print(str(np.mean(cov)))

plt.tight_layout()
fig.savefig("2_3_training_beta_dist.png")
plt.show()