import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet 
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as ari


K = 3 # サイン総数
# データセット読み込み
x_nd_1 = np.load("./dataset/data2d_1.npy")
x_nd_2 = np.load("./dataset/data2d_2.npy")
z_truth_n = np.load("./dataset/true_label2d.npy") 
D = len(x_nd_1);dim = len(x_nd_1[0]) # データ総数

###### ここから事前分布のパラメータを設定 ######
# muの事前分布のパラメータを指定
beta = 1.0
m_d_1 = np.repeat(0.0, dim)
m_d_2 = np.repeat(0.0, dim)

# lambdaの事前分布のパラメータを指定
w_dd_1 = np.identity(dim) * 0.01
w_dd_2 = np.identity(dim) * 0.01
nu = dim

#\mu, \lambdaの初期値を決定
mu_kd_1 = np.empty((K, dim))
lambda_kdd_1 = np.empty((K, dim, dim))
mu_kd_2 = np.empty((K, dim))
lambda_kdd_2 = np.empty((K, dim, dim))
for k in range(K):
    # クラスタkの精度行列をサンプル
    lambda_kdd_1[k] = wishart.rvs(df=nu, scale=w_dd_1, size=1)
    lambda_kdd_2[k] = wishart.rvs(df=nu, scale=w_dd_2, size=1)
    
    # クラスタkの平均をサンプル
    mu_kd_1[k] = np.random.multivariate_normal(
        mean=m_d_1, cov=np.linalg.inv(beta * lambda_kdd_1[k])
    ).flatten()
    mu_kd_2[k] = np.random.multivariate_normal(
        mean=m_d_2, cov=np.linalg.inv(beta * lambda_kdd_2[k])
    ).flatten()

# パラメータを初期化
eta_nk = np.zeros((D, K))
z_nk = np.zeros((D, K))
beta_hat_k_1 = np.zeros(K)
beta_hat_k_2 = np.zeros(K)
m_hat_kd_1 = np.zeros((K, dim))
m_hat_kd_2 = np.zeros((K, dim))
w_hat_kdd_1 = np.zeros((K, dim, dim))
w_hat_kdd_2 = np.zeros((K, dim, dim))
nu_hat_k_1 = np.zeros(K)
nu_hat_k_2 = np.zeros(K)

# 推移の確認用の変数
trace_z_in = [np.repeat(np.nan, D)]
trace_mu_ikd_1 = [mu_kd_1.copy()]
trace_mu_ikd_2 = [mu_kd_2.copy()]
trace_lambda_ikdd_1 = [lambda_kdd_1.copy()]
trace_lambda_ikdd_2 = [lambda_kdd_2.copy()]
trace_beta_ik_1 = [np.repeat(beta, K)]
trace_beta_ik_2 = [np.repeat(beta, K)]
trace_m_ikd_1 = [np.repeat(m_d_1.reshape((1, dim)), K, axis=0)]
trace_m_ikd_2 = [np.repeat(m_d_2.reshape((1, dim)), K, axis=0)]
trace_w_ikdd_1 = [np.repeat(w_dd_1.reshape((1, dim, dim)), K, axis=0)]
trace_w_ikdd_2 = [np.repeat(w_dd_2.reshape((1, dim, dim)), K, axis=0)]
trace_nu_ik_1 = [np.repeat(nu, K)]
trace_nu_ik_2 = [np.repeat(nu, K)]

z_nk = np.random.multinomial(1, [1/K]*K, size=D) # zの初期化
###### ここまで事前分布のパラメータを設定 ######

###### ここからギブスサンプリング ######
iteration = 100 # ギブスサンプリングの試行回数を指定
ARI = np.zeros((iteration)) # イテレーション毎のARIを格納する変数

# ギブスサンプリング
for i in range(iteration):
    print(f"----------------------{i+1}試行目------------------------")
    z_pred_n = [] # モデルの予測したカテゴリ
    
    # \mu, \lambdaのサンプリングに関して
    for k in range(K):
        # muの事後分布のパラメータを計算：ベイズ推論式(4.99)
        beta_hat_k_1[k] = np.sum(z_nk[:, k]) + beta; beta_hat_k_2[k] = np.sum(z_nk[:, k]) + beta
        m_hat_kd_1[k] = np.sum(z_nk[:, k] * x_nd_1.T, axis=1); m_hat_kd_2[k] = np.sum(z_nk[:, k] * x_nd_2.T, axis=1)
        m_hat_kd_1[k] += beta * m_d_1; m_hat_kd_2[k] += beta * m_d_2
        m_hat_kd_1[k] /= beta_hat_k_1[k]; m_hat_kd_2[k] /= beta_hat_k_2[k]
        
        # lambdaの事後分布のパラメータを計算：ベイズ推論式(4.103)
        tmp_w_dd_1 = np.dot((z_nk[:, k] * x_nd_1.T), x_nd_1); tmp_w_dd_2 = np.dot((z_nk[:, k] * x_nd_2.T), x_nd_2)
        tmp_w_dd_1 += beta * np.dot(m_d_1.reshape(dim, 1), m_d_1.reshape(1, dim)); tmp_w_dd_2 += beta * np.dot(m_d_2.reshape(dim, 1), m_d_2.reshape(1, dim))
        tmp_w_dd_1 -= beta_hat_k_1[k] * np.dot(m_hat_kd_1[k].reshape(dim, 1), m_hat_kd_1[k].reshape(1, dim))
        tmp_w_dd_2 -= beta_hat_k_2[k] * np.dot(m_hat_kd_2[k].reshape(dim, 1), m_hat_kd_2[k].reshape(1, dim))
        tmp_w_dd_1 += np.linalg.inv(w_dd_1); tmp_w_dd_2 += np.linalg.inv(w_dd_2)
        w_hat_kdd_1[k] = np.linalg.inv(tmp_w_dd_1); w_hat_kdd_2[k] = np.linalg.inv(tmp_w_dd_2)
        nu_hat_k_1[k] = np.sum(z_nk[:, k]) + nu
        nu_hat_k_2[k] = np.sum(z_nk[:, k]) + nu
        
        # lambdaをサンプル：ベイズ推論式(4.102)
        lambda_kdd_1[k] = wishart.rvs(size=1, df=nu_hat_k_1[k], scale=w_hat_kdd_1[k])
        lambda_kdd_2[k] = wishart.rvs(size=1, df=nu_hat_k_2[k], scale=w_hat_kdd_2[k])
        
        # muをサンプル：ベイズ推論式(4.98)
        mu_kd_1[k] = np.random.multivariate_normal(
            mean=m_hat_kd_1[k], cov=np.linalg.inv(beta_hat_k_1[k] * lambda_kdd_1[k]), size=1
        ).flatten()
        mu_kd_2[k] = np.random.multivariate_normal(
            mean=m_hat_kd_2[k], cov=np.linalg.inv(beta_hat_k_2[k] * lambda_kdd_2[k]), size=1
        ).flatten()
    
    # zのサンプリングに関して
    # 潜在変数の事後分布のパラメータを計算:ベイズ推論式(4.94)
    for k in range(K):
        tmp_eta_n = np.diag(
            -0.5 * (x_nd_1 - mu_kd_1[k]).dot(lambda_kdd_1[k]).dot((x_nd_1 - mu_kd_1[k]).T)
        ).copy() 
        tmp_eta_n += np.diag(
            -0.5 * (x_nd_2 - mu_kd_2[k]).dot(lambda_kdd_2[k]).dot((x_nd_2 - mu_kd_2[k]).T)
        ).copy() 
        tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd_1[k]) + 1e-7)
        tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd_2[k]) + 1e-7)
        # tmp_eta_n += np.log(pi_k[k] + 1e-7) # ベイズ推論式(4.92)の混合比に関する部分
        eta_nk[:, k] = np.exp(tmp_eta_n)
    eta_nk /= np.sum(eta_nk, axis=1, keepdims=True) # 正規化
    
    
    # 潜在変数をサンプル：ベイズ推論式(4.93)
    for d in range(D):
        z_nk[d] = np.random.multinomial(n=1, pvals=eta_nk[d], size=1).flatten()
        z_pred_n.append(np.argmax(z_nk[d]))
    
    ARI[i] = np.round(ari(z_truth_n,z_pred_n),3)
    print(f"ARI:{ARI[i]}")

    # 値を記録
    _, z_n = np.where(z_nk == 1)
    trace_z_in.append(z_n.copy())
    trace_mu_ikd_1.append(mu_kd_1.copy())
    trace_lambda_ikdd_1.append(lambda_kdd_1.copy())
    trace_beta_ik_1.append(beta_hat_k_1.copy())
    trace_m_ikd_1.append(m_hat_kd_1.copy())
    trace_w_ikdd_1.append(w_hat_kdd_1.copy())
    trace_nu_ik_1.append(nu_hat_k_1.copy())

plt.plot(range(0,iteration), ARI, marker="None")
plt.xlabel('iteration')
plt.ylabel('ARI')
plt.ylim(0,1)
#plt.savefig("ari.png")
plt.show()