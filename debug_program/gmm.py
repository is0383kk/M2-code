import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet # 多次元ガウス分布, ウィシャート分布, ディリクレ分布
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as ari


K = 4
x_nd = np.loadtxt("./dataset/data2.txt")
#x_nd = np.loadtxt("./dataset/data2.txt")
z_truth_n = np.loadtxt("./dataset/true_label.txt") 
# データ総数
D = len(x_nd)
print(f"データ総数:{len(x_nd)}")
# 次元数を設定:(固定)
dim = len(x_nd[0])
print(f"データ次元数:{len(x_nd[0])}")
# クラスタ数を指定

#事前分布のパラメータ
# muの事前分布のパラメータを指定
beta = 1.0
m_d = np.repeat(0.0, dim)

# lambdaの事前分布のパラメータを指定
w_dd = np.identity(dim) * 0.02 # 人工データのときは0.05
#print("dim",dim)
nu = dim

#\mu, \lambda, \piの初期値を決定
# 観測モデルのパラメータをサンプル
mu_kd = np.empty((K, dim))
lambda_kdd = np.empty((K, dim, dim))

for k in range(K):
    # クラスタkの精度行列をサンプル
    lambda_kdd[k] = wishart.rvs(df=nu, scale=w_dd, size=1)
    #print("lambda_kdd",lambda_kdd[k])
    
    # クラスタkの平均をサンプル
    mu_kd[k] = np.random.multivariate_normal(
        mean=m_d, cov=np.linalg.inv(beta * lambda_kdd[k])
    ).flatten()

# パラメータを初期化
eta_nk = np.zeros((D, K))
s_nk = np.zeros((D, K))
beta_hat_k = np.zeros(K)
m_hat_kd = np.zeros((K, dim))
w_hat_kdd = np.zeros((K, dim, dim))
nu_hat_k = np.zeros(K)

# 推移の確認用の受け皿を作成
trace_s_in = [np.repeat(np.nan, D)]
trace_mu_ikd = [mu_kd.copy()]
trace_lambda_ikdd = [lambda_kdd.copy()]
trace_beta_ik = [np.repeat(beta, K)]
trace_m_ikd = [np.repeat(m_d.reshape((1, dim)), K, axis=0)]
trace_w_ikdd = [np.repeat(w_dd.reshape((1, dim, dim)), K, axis=0)]
trace_nu_ik = [np.repeat(nu, K)]


# 試行回数を指定
iteration = 100
ARI = np.zeros((iteration))


# ギブスサンプリング
for i in range(iteration):
    print(f"{i+1}試行目")
    pred_label = []
    """zのサンプリング"""
    # 潜在変数の事後分布のパラメータを計算:式(4.94)
    for k in range(K):
        tmp_eta_n = np.diag(-0.5 * (x_nd - mu_kd[k]).dot(lambda_kdd[k]).dot((x_nd - mu_kd[k]).T)).copy()
        tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
        eta_nk[:, k] = np.exp(tmp_eta_n)
    eta_nk /= np.sum(eta_nk, axis=1, keepdims=True) # 正規化
    
    # 潜在変数をサンプル：式(4.93)
    for d in range(D):
        s_nk[d] = np.random.multinomial(n=1, pvals=eta_nk[d], size=1).flatten()
        pred_label.append(np.argmax(s_nk[d]))

    # 観測モデルのパラメータをサンプル
    for k in range(K):
        # muの事後分布のパラメータを計算：式(4.99)
        beta_hat_k[k] = np.sum(s_nk[:, k]) + beta
        m_hat_kd[k] = np.sum(s_nk[:, k] * x_nd.T, axis=1)
        m_hat_kd[k] += beta * m_d
        m_hat_kd[k] /= beta_hat_k[k]        
        # lambdaの事後分布のパラメータを計算：式(4.103)
        tmp_w_dd = np.dot((s_nk[:, k] * x_nd.T), x_nd)
        tmp_w_dd += beta * np.dot(m_d.reshape(dim, 1), m_d.reshape(1, dim))
        tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(dim, 1), m_hat_kd[k].reshape(1, dim))
        tmp_w_dd += np.linalg.inv(w_dd)
        w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
        nu_hat_k[k] = np.sum(s_nk[:, k]) + nu # 自由度
                
        # lambdaをサンプル：式(4.102)
        lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
        # muをサンプル：式(4.98)
        mu_kd[k] = np.random.multivariate_normal(
            mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]), size=1
        ).flatten()
        
    
    #ARI[i] = np.round(calc_ari(pred_label,z_truth_n)[0],3)
    #print(f"ARI:{ARI[i]}")
    ARI[i] = np.round(ari(z_truth_n,pred_label),3)
    print(f"ARI:{ARI[i]}")


    # 値を記録
    _, s_n = np.where(s_nk == 1)
    trace_s_in.append(s_n.copy())
    trace_mu_ikd.append(mu_kd.copy())
    trace_lambda_ikdd.append(lambda_kdd.copy())
    trace_beta_ik.append(beta_hat_k.copy())
    trace_m_ikd.append(m_hat_kd.copy())
    trace_w_ikdd.append(w_hat_kdd.copy())
    trace_nu_ik.append(nu_hat_k.copy())


plt.plot(range(0,iteration),ARI,marker="None")
plt.xlabel('iteration')
plt.ylabel('ARI')
plt.savefig("ari.png")