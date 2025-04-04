import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

random_seed = 25
np.random.seed(random_seed)

# Define the mean function and the data generating function

def f(x, a, b):
    return a/(1+(x-b)**2)

def mu(x):
    return f(x, 1, 0) + f(x, 2, 3)

def generate_data(n):
    x = np.random.uniform(-5, 5, n)
    y = mu(x) + 0.5*np.random.randn(n)
    return np.column_stack((x, y)) # return matrix with X, Y in first/second column

# Generate proper training, calibration and test data

n_train, n_cal, n_test = 1000, 3000, 3000
n = n_train + n_cal + n_test
alpha = 0.1

D = generate_data(n)

D_train = D[:n_train,:]
D_cal = D[n_train:(n_train + n_cal),:]
D_test = D[(n_train + n_cal):(n_train + n_cal + n_test),:]

X_train, Y_train = D_train[:, 0], D_train[:, 1]
X_cal, Y_cal = D_cal[:, 0], D_cal[:, 1]
X_test, Y_test = D_test[:, 0], D_test[:, 1]

# Define the conformity score
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

# Generate prediction interval
Y_pred_LR = model_LR.predict(X_test.reshape(-1, 1)).flatten()
y_upper_LR = Y_pred_LR + q_LR
y_lower_LR = Y_pred_LR - q_LR

# Hyperparameters for random forest model
n_estimators = [100, 150, 200]
max_depth = [5, 10, 15]
min_samples_split = [2, 6, 10]

param_grid = {"n_estimators": n_estimators,
              "max_depth": max_depth,
              "min_samples_split": min_samples_split}

model_RF = GridSearchCV(estimator = RandomForestRegressor(random_state=random_seed),
                     param_grid= param_grid,
                     cv = 5,
                     n_jobs=4)

# Fit the random forest
model_RF.fit(X_train.reshape(-1, 1), Y_train.ravel())

# Calibrate
scores_RF = absolute_residual_score(X_cal.reshape(-1, 1), Y_cal, model_RF)
q_RF = np.sort(scores_RF.flatten())[q_cal - 1]

# Predict
Y_pred_RF = model_RF.predict(X_test.reshape(-1, 1)).flatten()
y_upper_RF = Y_pred_RF + q_RF
y_lower_RF = Y_pred_RF - q_RF

# Check coverage and interval length
def check_coverage(alpha, n, y, y_lower, y_upper, q_cal):
    if q_cal >= n+1:
        return 1
    else:
        return np.mean((y <= y_upper) & (y >= y_lower))

LR_cov = check_coverage(alpha, n_cal, Y_test, y_lower_LR, y_upper_LR, q_cal)
RF_cov = check_coverage(alpha, n_cal, Y_test, y_lower_RF, y_upper_RF, q_cal)
LR_length = np.mean(np.abs(y_upper_LR - y_lower_LR))
RF_length = np.mean(np.abs(y_upper_RF - y_lower_RF))

print("Linear regression coverage: " + str(LR_cov))
print("Random forest coverage: " + str(RF_cov))
print("Linear regression average length: " + str(LR_length))
print("Random forest average length: " + str(RF_length))

# Plot random forest prediction interval

plt.rcParams['text.usetex'] = True

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(12, 6))

ax1.set_xlabel(r"$x$")
ax2.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax2.set_ylabel(r"$y$")

ax1.scatter(X_test, Y_test, color="blue", label="Test data")
ax1.plot(X_test, Y_pred_LR, color="black", linestyle="dashed", label=r"$\hat{\mu}_{LR}(x)$")
ax1.fill_between(X_test[np.argsort(X_test)], y_lower_LR[np.argsort(X_test)], y_upper_LR[np.argsort(X_test)], color="red", alpha = 0.5, label="Prediction interval")
ax1.legend()

ax2.scatter(X_test, Y_test, color="blue", label="Test data")
ax2.plot(X_test[np.argsort(X_test)], Y_pred_RF[np.argsort(X_test)], color="black", linestyle="dashed", label=r"$\hat{\mu}_{RF}(x)$")
ax2.fill_between(X_test[np.argsort(X_test)], y_lower_RF[np.argsort(X_test)], y_upper_RF[np.argsort(X_test)], color="red", alpha = 0.5, label="Prediction interval")
ax2.legend()

fig.savefig("figures/linearreg_RF_example.png")
plt.show()