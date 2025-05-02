import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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

R = 2000 # number of calibration/test sets we average over
n_train, n_cal, n_test = 1000, 1500, 1500
n = n_cal + n_test

D_train = generate_data(n_train)
X_train, Y_train = D_train[:, 0], D_train[:, 1]

alpha = 0.1
q_cal = int(np.ceil((1-alpha)*(n_cal + 1)))

# Linear regression fit ------------------------------------------------------

model_LR = LinearRegression()
model_LR.fit(X_train.reshape(-1,1), Y_train.ravel())

# Random forests fit ---------------------------------------------------------
n_estimators = [100, 150, 200]
max_depth = [5, 10, 15]
min_samples_split = [2, 6, 10]

params= {"n_estimators": n_estimators,
              "max_depth": max_depth,
              "min_samples_split": min_samples_split}

model_RF = GridSearchCV(estimator = RandomForestRegressor(),
                     param_grid= params,
                     cv = 5,
                     n_jobs=4)

model_RF.fit(X_train.reshape(-1, 1), Y_train.ravel())

cov_LR = np.zeros((R, ))
lens_LR = np.zeros((R, ))
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
    lens_LR[r] = np.mean(y_upper_LR - y_lower_LR)

cov_RF = np.zeros((R, ))
lens_RF = np.zeros((R, ))
for r in range(R):
    D = generate_data(n)
    D_cal = D[:n_cal,:]
    D_test = D[n_cal:(n_cal + n_test),:]
    X_cal, Y_cal = D_cal[:, 0], D_cal[:, 1]
    X_test_RF, Y_test_RF = D_test[:, 0], D_test[:, 1]

    scores_RF = absolute_residual_score(X_cal.reshape(-1, 1), Y_cal, model_RF)
    Y_pred_RF = model_RF.predict(X_test_RF.reshape(-1, 1)).flatten()
    q_RF = np.sort(scores_RF.flatten())[q_cal - 1]
    y_upper_RF = Y_pred_RF + q_RF
    y_lower_RF = Y_pred_RF - q_RF

    cov_RF[r] = check_coverage(alpha, n_cal, Y_test_RF, y_lower_RF, y_upper_RF, q_cal)
    lens_RF[r] = np.mean(y_upper_RF - y_lower_RF)

print(str(np.mean(cov_LR)))
print(str(np.mean(cov_RF)))
print(str(np.mean(lens_LR)))
print(str(np.mean(lens_RF)))

fig1, ax1 = plt.subplots()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.scatter(X_test_LR, Y_test_LR, color="blue", label="Test data")
ax1.plot(X_test_LR[np.argsort(X_test_LR)], Y_pred_LR[np.argsort(X_test_LR)], color="black", linestyle="dashed", label=r"$\hat{\mu}(x)$")
ax1.fill_between(X_test_LR[np.argsort(X_test_LR)], y_lower_LR[np.argsort(X_test_LR)], y_upper_LR[np.argsort(X_test_LR)], color="red", alpha = 0.5, label="Prediction interval")
ax1.legend()
plt.show()
fig1.savefig("2_3_LR.png")

fig2, ax2 = plt.subplots()
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.scatter(X_test_RF, Y_test_RF, color="blue", label="Test data")
ax2.plot(X_test_RF[np.argsort(X_test_RF)], Y_pred_RF[np.argsort(X_test_RF)], color="black", linestyle="dashed", label=r"$\hat{\mu}(x)$")
ax2.fill_between(X_test_RF[np.argsort(X_test_RF)], y_lower_RF[np.argsort(X_test_RF)], y_upper_RF[np.argsort(X_test_RF)], color="red", alpha = 0.5, label="Prediction interval")
ax2.legend()
plt.show()
fig2.savefig("2_3_RF.png")

fig3, ax3 = plt.subplots()
ax3.set_xlabel("Conformity scores")
ax3.set_ylabel("Frequency")
ax3.hist(scores_LR, bins=np.arange(0, 2.75, 0.25), label=r"$\hat{s}(X_i, Y_i)$")
ax3.axvline(q_LR, color="orange", linestyle="dashed", label=r"$\hat{Q}_{(\hat{S}_1, \ldots, \hat{S}_n, \infty)}(1-\alpha)$")
ax3.legend()
fig3.savefig("2_3_LR_scores.png")

fig4, ax4 = plt.subplots()
ax4.set_xlabel("Conformity scores")
ax4.set_ylabel("Frequency")
ax4.hist(scores_RF, bins=np.arange(0, 2.75, 0.25), label=r"$\hat{s}(X_i, Y_i)$")
ax4.axvline(q_RF, color="orange", linestyle="dashed", label=r"$\hat{Q}_{(\hat{S}_1, \ldots, \hat{S}_n, \infty)}(1-\alpha)$")
ax4.legend()
fig4.savefig("2_3_RF_scores.png")

fig5, ax5 = plt.subplots()
ax5.set_xlabel("Average interval lengths")
ax5.set_ylabel("Density")
ax5.hist(lens_LR, bins=20, alpha = 0.5, density=True, label="Linear regression")
ax5.hist(lens_RF, bins=20, alpha = 0.5, density=True, label="Random forests")
ax5.legend()
plt.show()
fig5.savefig("2_3_lens.png")