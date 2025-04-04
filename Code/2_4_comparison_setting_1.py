import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.base import clone

random_seed = 7
np.random.seed(random_seed)

# Setting 3 -------------- Heteroscedastic + outliers ------------------------

def mu(x, a, b):
    return a + b*x

def generate_data(a, b, n):
    x = np.random.uniform(-5, 5, n)
    s = 2
    y = mu(x, a, b) + s*np.random.randn(n)
    return np.column_stack((x, y))

n_train, n_cal, n_test = 1000, 3000, 3000
n = n_train + n_cal + n_test
alpha = 0.1

D = generate_data(1, -1, n)

D_train = D[:n_train,:]
D_cal = D[n_train:(n_train + n_cal),:]
D_test = D[(n_train + n_cal):(n_train + n_cal + n_test),:]

X_train, Y_train = D_train[:, 0], D_train[:, 1]
X_cal, Y_cal = D_cal[:, 0], D_cal[:, 1]
X_test, Y_test = D_test[:, 0], D_test[:, 1]

def absolute_residual_score(x, y, model):
    return np.abs(y - model.predict(x).flatten())

def locally_weighted_score(x, y, mu_model, sigma_model):
    return np.abs(y - mu_model.predict(x).flatten())/sigma_model.predict(x).flatten()

def quantile_score(x, y, lower_q_model, upper_q_model):
    l = lower_q_model.predict(x) - y
    u = y - upper_q_model.predict(x)
    return np.maximum(l, u)

def check_coverage(alpha, n, y, y_lower, y_upper, q_cal):
    if q_cal >= n+1:
        return 1
    else:
        return np.mean((y <= y_upper) & (y >= y_lower))

q_cal = int(np.ceil((1-alpha)*(n_cal + 1)))

# Split CP with random forests
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

model_RF.fit(X_train.reshape(-1, 1), Y_train.ravel())

scores_RF = absolute_residual_score(X_cal.reshape(-1, 1), Y_cal, model_RF)

Y_pred_RF = model_RF.predict(X_test.reshape(-1, 1)).flatten()

q_absolute = np.sort(scores_RF.flatten())[q_cal - 1]
y_upper_RF = Y_pred_RF + q_absolute
y_lower_RF = Y_pred_RF - q_absolute

# Locally weighted

abs_errors = absolute_residual_score(X_train.reshape(-1, 1), Y_train, model_RF)

sigma_model = GridSearchCV(estimator = RandomForestRegressor(random_state=random_seed),
                     param_grid= param_grid,
                     cv = 5,
                     n_jobs=4)
sigma_model.fit(X_train.reshape(-1, 1), abs_errors.ravel())

weighted_scores = locally_weighted_score(X_cal.reshape(-1, 1), Y_cal, model_RF, sigma_model)

sigma_pred = sigma_model.predict(X_test.reshape(-1,1)).flatten()

q_weighted = np.sort(weighted_scores.flatten())[q_cal - 1]

y_upper_W = Y_pred_RF + q_weighted * sigma_pred
y_lower_W = Y_pred_RF - q_weighted * sigma_pred

# CQR

param_grid = {
    "n_estimators": [100, 150, 200, 250],
    "max_depth": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 5, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 20, 30, 50]
}
q = 0.05
neg_mean_pinball_loss_05p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=q,
    greater_is_better=False,  # maximize the negative loss
)
qrf = RandomForestQuantileRegressor(q=q, random_state=random_seed)
search_05p = RandomizedSearchCV(
    qrf,
    param_grid,
    n_iter=25,  # increase this if computational budget allows
    scoring=neg_mean_pinball_loss_05p_scorer,
    n_jobs=4,
    random_state=random_seed,
).fit(X_train.reshape(-1,1), Y_train.ravel())

q = 0.95
neg_mean_pinball_loss_95p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=q,
    greater_is_better=False,  # maximize the negative loss
)
search_95p = clone(search_05p).set_params(
    estimator__q=q,
    scoring=neg_mean_pinball_loss_95p_scorer,
)
search_95p.fit(X_train.reshape(-1, 1), Y_train.ravel())

quantile_scores = quantile_score(X_cal.reshape(-1, 1), Y_cal, search_05p, search_95p)
q_quantile = np.sort(quantile_scores.flatten())[q_cal - 1]

Y_lower_pred_Q = search_05p.predict(X_test.reshape(-1, 1))
Y_upper_pred_Q = search_95p.predict(X_test.reshape(-1, 1))
y_lower_Q = Y_lower_pred_Q - q_quantile
y_upper_Q = Y_upper_pred_Q + q_quantile

split_cov = check_coverage(alpha, n_cal, Y_test, y_lower_RF, y_upper_RF, q_cal)
LW_cov = check_coverage(alpha, n_cal, Y_test, y_lower_W, y_upper_W, q_cal)
CQR_cov = check_coverage(alpha, n_cal, Y_test, y_lower_Q, y_upper_Q, q_cal)

split_len = np.mean(y_upper_RF - y_lower_RF)
LW_len = np.mean(y_upper_W - y_lower_W)
CQR_len = np.mean(y_upper_Q - y_lower_Q)

print("Split conformal coverage: " + str(split_cov))
print("Split conformal length: " + str(split_len))
print("LW conformal coverage: " + str(LW_cov))
print("LW conformal length: " + str(LW_len))
print("CQR conformal coverage: " + str(CQR_cov))
print("CQR conformal length: " + str(CQR_len))

fig1, ax1 = plt.subplots()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.scatter(X_test, Y_test, color="blue", label="Test data")
ax1.plot(X_test[np.argsort(X_test)], Y_pred_RF[np.argsort(X_test)], color="black", linestyle="dashed", label=r"$\hat{\mu}(x)$")
ax1.fill_between(X_test[np.argsort(X_test)], y_lower_RF[np.argsort(X_test)], y_upper_RF[np.argsort(X_test)], color="red", alpha = 0.5, label="Prediction interval")
ax1.legend()
fig1.savefig("figures/2_4_homoscedastic_RF.png")
plt.show()

fig2, ax2 = plt.subplots()
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.scatter(X_test, Y_test, color="blue", label="Test data")
ax2.plot(X_test[np.argsort(X_test)], Y_pred_RF[np.argsort(X_test)], color="black", linestyle="dashed", label=r"$\hat{\mu}(x)$")
ax2.fill_between(X_test[np.argsort(X_test)], y_lower_W[np.argsort(X_test)], y_upper_W[np.argsort(X_test)], color="red", alpha = 0.5, label="Prediction interval")
ax2.legend()
fig2.savefig("figures/2_4_homoscedastic_LW.png")
plt.show()

fig3, ax3 = plt.subplots()
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$y$")
ax3.scatter(X_test, Y_test, color="blue", label="Test data")
ax3.plot(X_test[np.argsort(X_test)], Y_lower_pred_Q[np.argsort(X_test)], color="black", linestyle="dashed", label=r"$\hat{q}_{\alpha/2}(x)$")
ax3.plot(X_test[np.argsort(X_test)], Y_upper_pred_Q[np.argsort(X_test)], color="black", linestyle="dashed", label=r"$\hat{q}_{1 - \alpha/2}(x)$")
ax3.fill_between(X_test[np.argsort(X_test)], y_lower_W[np.argsort(X_test)], y_upper_W[np.argsort(X_test)], color="red", alpha = 0.5, label="Prediction interval")
ax3.legend()
fig3.savefig("figures/2_4_homoscedastic_CQR.png")
plt.show()