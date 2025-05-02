import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from quantile_forest import RandomForestQuantileRegressor

# Setting 1 -------------- Homoscedastic ------------------------
def mu(x, a, b):
    return a + b*x

def generate_data(a, b, n):
    x = np.random.uniform(-5, 5, n)
    s = 2
    y = mu(x, a, b) + s*np.random.randn(n)
    return np.column_stack((x, y))

def absolute_residual_score(x, y, model):
    return np.abs(y - model.predict(x).flatten())

def locally_weighted_score(x, y, mu_model, sigma_model):
    return np.abs(y - mu_model.predict(x).flatten())/sigma_model.predict(x).flatten()

def quantile_score(x, y, q_model):
    model_pred = q_model.predict(x)
    l = model_pred[:, 0] - y
    u = y - model_pred[:, 1]
    return np.maximum(l, u)

def check_coverage(alpha, n, y, y_lower, y_upper, q_cal):
    if q_cal >= n+1:
        return 1
    else:
        return np.mean((y <= y_upper) & (y >= y_lower))

R = 2000 # number of calibration/test sets we average over
n_train, n_cal, n_test = 1000, 1500, 1500
n = n_cal + n_test

D_train = generate_data(1, -1, n_train)
X_train, Y_train = D_train[:, 0], D_train[:, 1]

alpha = 0.1
q_cal = int(np.ceil((1-alpha)*(n_cal + 1)))

# Absolute residual score training ---------------------------

n_estimators = [100, 150, 200]
max_depth = [5, 10, 15]
min_samples_split = [2, 5, 8, 15]

params= {"n_estimators": n_estimators,
              "max_depth": max_depth,
              "min_samples_split": min_samples_split}

model_RF = GridSearchCV(estimator = RandomForestRegressor(),
                     param_grid= params,
                     cv = 5,
                     n_jobs=4)

model_RF.fit(X_train.reshape(-1, 1), Y_train.ravel())

# Locally weighted training ---------------------------------

abs_errors = absolute_residual_score(X_train.reshape(-1, 1), Y_train, model_RF)

sigma_model = GridSearchCV(estimator = RandomForestRegressor(),
                     param_grid= params,
                     cv = 5,
                     n_jobs=4)
sigma_model.fit(X_train.reshape(-1, 1), abs_errors.ravel())

# CQR training ----------------------------------------------

model_Q = GridSearchCV(estimator = RandomForestQuantileRegressor(default_quantiles=[alpha/2, 1-alpha/2]),
                     param_grid= params,
                     cv = 5, 
                     n_jobs=4)
model_Q.fit(X_train.reshape(-1,1), Y_train.ravel())

#-------------------------------------------------------------

cov_RF = np.zeros((R, ))
lens_RF = np.zeros((R, ))
for r in range(R):
    D = generate_data(1, -1, n)
    D_cal = D[:n_cal,:]
    D_test = D[n_cal:(n_cal + n_test),:]
    X_cal, Y_cal = D_cal[:, 0], D_cal[:, 1]
    X_test_RF, Y_test_RF = D_test[:, 0], D_test[:, 1]

    scores_RF = absolute_residual_score(X_cal.reshape(-1, 1), Y_cal, model_RF)
    Y_pred_RF = model_RF.predict(X_test_RF.reshape(-1, 1)).flatten()
    q_absolute = np.sort(scores_RF.flatten())[q_cal - 1]
    y_upper_RF = Y_pred_RF + q_absolute
    y_lower_RF = Y_pred_RF - q_absolute

    cov_RF[r] = check_coverage(alpha, n_cal, Y_test_RF, y_lower_RF, y_upper_RF, q_cal)
    lens_RF[r] = np.mean(y_upper_RF - y_lower_RF)

cov_LW = np.zeros((R, ))
lens_LW = np.zeros((R, ))
for r in range(R):
    D = generate_data(1, -1, n)
    D_cal = D[:n_cal,:]
    D_test = D[n_cal:(n_cal + n_test),:]
    X_cal, Y_cal = D_cal[:, 0], D_cal[:, 1]
    X_test_LW, Y_test_LW = D_test[:, 0], D_test[:, 1]

    scores_LW = locally_weighted_score(X_cal.reshape(-1, 1), Y_cal, model_RF, sigma_model)
    Y_pred_LW = model_RF.predict(X_test_LW.reshape(-1, 1)).flatten()
    sigma_pred = sigma_model.predict(X_test_LW.reshape(-1,1)).flatten()

    q_weighted = np.sort(scores_LW.flatten())[q_cal - 1]
    y_upper_W = Y_pred_LW + q_weighted * sigma_pred
    y_lower_W = Y_pred_LW - q_weighted * sigma_pred

    cov_LW[r] = check_coverage(alpha, n_cal, Y_test_LW, y_lower_W, y_upper_W, q_cal)
    lens_LW[r] = np.mean(y_upper_W - y_lower_W)

cov_Q = np.zeros((R, ))
cov_base_Q = np.zeros((R, ))
lens_Q = np.zeros((R,))
lens_base_Q = np.zeros((R, ))
for r in range(R):
    D = generate_data(1, -1, n)
    D_cal = D[:n_cal,:]
    D_test = D[n_cal:(n_cal + n_test),:]
    X_cal, Y_cal = D_cal[:, 0], D_cal[:, 1]
    X_test_Q, Y_test_Q = D_test[:, 0], D_test[:, 1]

    scores_Q = quantile_score(X_cal.reshape(-1,1), Y_cal, model_Q)
    Y_pred_Q = model_Q.predict(X_test_Q.reshape(-1, 1))
    Y_pred_Q_upper = Y_pred_Q[:, 1]
    Y_pred_Q_lower = Y_pred_Q[:, 0]

    q_quantile = np.sort(scores_Q.flatten())[q_cal - 1]
    y_upper_Q = Y_pred_Q_upper + q_quantile
    y_lower_Q = Y_pred_Q_lower - q_quantile

    cov_Q[r] = check_coverage(alpha, n_cal, Y_test_Q, y_lower_Q, y_upper_Q, q_cal)
    lens_Q[r] = np.mean(y_upper_Q-y_lower_Q)

    cov_base_Q[r] = check_coverage(alpha, n_cal, Y_test_Q, Y_pred_Q_lower, Y_pred_Q_upper, q_cal)
    lens_base_Q[r] = np.mean(Y_pred_Q_upper - Y_pred_Q_lower)

split_cov = np.mean(cov_RF)
LW_cov = np.mean(cov_LW)
Q_cov = np.mean(cov_Q)
base_Q_cov = np.mean(cov_base_Q)

split_len = np.mean(lens_RF)
LW_len = np.mean(lens_LW)
Q_len = np.mean(lens_Q)
base_Q_len = np.mean(lens_base_Q)

print(str(split_cov))
print(str(LW_cov))
print(str(Q_cov))
print(str(base_Q_cov))

print(str(split_len))
print(str(LW_len))
print(str(Q_len))
print(str(base_Q_len))

fig1, ax1 = plt.subplots()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.scatter(X_test_RF, Y_test_RF, color="blue", label="Test data")
ax1.plot(X_test_RF[np.argsort(X_test_RF)], Y_pred_RF[np.argsort(X_test_RF)], color="black", linestyle="dashed", label=r"$\hat{\mu}(x)$")
ax1.fill_between(X_test_RF[np.argsort(X_test_RF)], y_lower_RF[np.argsort(X_test_RF)], y_upper_RF[np.argsort(X_test_RF)], color="red", alpha = 0.5, label="Prediction interval")
ax1.legend()
plt.show()
fig1.savefig("2_4_homoscedastic_RF.png")

fig2, ax2 = plt.subplots()
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.scatter(X_test_LW, Y_test_LW, color="blue", label="Test data")
ax2.plot(X_test_LW[np.argsort(X_test_LW)], Y_pred_LW[np.argsort(X_test_LW)], color="black", linestyle="dashed", label=r"$\hat{\mu}(x)$")
ax2.fill_between(X_test_LW[np.argsort(X_test_LW)], y_lower_W[np.argsort(X_test_LW)], y_upper_W[np.argsort(X_test_LW)], color="red", alpha = 0.5, label="Prediction interval")
ax2.legend()
plt.show()
fig2.savefig("2_4_homoscedastic_LW.png")

fig3, ax3 = plt.subplots()
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$y$")
ax3.scatter(X_test_Q, Y_test_Q, color="blue", label="Test data")
ax3.plot(X_test_Q[np.argsort(X_test_Q)], Y_pred_Q_lower[np.argsort(X_test_Q)], color="black", linestyle="dashed", label=r"$\hat{q}_{\alpha/2}(x)$")
ax3.plot(X_test_Q[np.argsort(X_test_Q)], Y_pred_Q_upper[np.argsort(X_test_Q)], color="black", linestyle="dashed", label=r"$\hat{q}_{1 - \alpha/2}(x)$")
ax3.fill_between(X_test_Q[np.argsort(X_test_Q)], y_lower_Q[np.argsort(X_test_Q)], y_upper_Q[np.argsort(X_test_Q)], color="red", alpha = 0.5, label="Prediction interval")
ax3.legend()
plt.show()
fig3.savefig("2_4_homoscedastic_CQR.png")


fig4, ax4 = plt.subplots()
ax4.set_xlabel("Average interval lengths")
ax4.set_ylabel("Density")
ax4.hist(lens_RF, bins=20, alpha = 0.5, density=True, label="Absolute residual")
ax4.hist(lens_LW, bins=20, alpha = 0.5, density=True, label="Locally weighted")
ax4.hist(lens_Q, bins=20, alpha = 0.5, density=True, label="Conformalised quantile regression")
ax4.legend()
plt.show()
fig4.savefig("2_4_lens_dist.png")