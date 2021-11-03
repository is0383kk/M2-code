import os
import numpy as np
from scipy.stats import wishart, multivariate_normal
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser(description='M-H algorithm M-GMM Example')
parser.add_argument('--sign', type=int, default=3, metavar='K', help='Number of sign')
parser.add_argument('--mode', type=int, default=-1, metavar='M', help='0:All reject, 1:ALL accept')
parser.add_argument('--iteration', type=int, default=100, metavar='N', help='number of iteration for MGMM_MH')
args = parser.parse_args()

# 保存用のディレクトリの作成
file_name = "debug"; model_dir = "./model"; dir_name = "./model/"+file_name# debugフォルダに保存される
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)

K = args.sign # サイン総数

# データセットの読み込み
c_nd_A = np.load("./dataset/data1d_1.npy") # Aの観測
c_nd_B = np.load("./dataset/data1d_2.npy") # Bの観測
z_truth_n = np.load("./dataset/true_label1d.npy") # 真のサインラベル

D = len(c_nd_A) # データ総数
dim = len(c_nd_A[0]) # データの次元数

############################## Initializing parameters ##############################
# ハイパーパラメータ
beta = 1.0; m_d_A = np.repeat(0.0, dim)
m_d_B = np.repeat(0.0, dim)
w_dd_A = np.identity(dim) * 0.1
w_dd_B = np.identity(dim) * 0.1 
nu = dim

# Initializing \mu, \Lambda
mu_kd_A = np.empty((K, dim))
lambda_kdd_A = np.empty((K, dim, dim))
mu_kd_B = np.empty((K, dim))
lambda_kdd_B = np.empty((K, dim, dim))
for k in range(K):
    lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1)
    lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
    mu_kd_A[k] = np.random.multivariate_normal(mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])).flatten()
    mu_kd_B[k] = np.random.multivariate_normal(mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])).flatten()

# 事前分布からサインの初期値を決定
w_dk_A = np.random.multinomial(1, [1/K]*K, size=D)
w_dk_B = np.random.multinomial(1, [1/K]*K, size=D)

# 各種パラメータの初期化
beta_hat_k_A = np.zeros(K)
beta_hat_k_B = np.zeros(K)
m_hat_kd_A = np.zeros((K, dim))
m_hat_kd_B = np.zeros((K, dim))
w_hat_kdd_A = np.zeros((K, dim, dim))
w_hat_kdd_B = np.zeros((K, dim, dim))
nu_hat_k_A = np.zeros(K)
nu_hat_k_B = np.zeros(K)
tmp_eta_nB = np.zeros((K, D))
eta_dkB = np.zeros((D, K))
tmp_eta_nA = np.zeros((K, D))
eta_dkA = np.zeros((D, K))
cat_liks_A = np.zeros(D)
cat_liks_B = np.zeros(D)


# 推移保存用
trace_w_in_A = [np.repeat(np.nan, D)]
trace_w_in_B = [np.repeat(np.nan, D)]
trace_mu_ikd_A = [mu_kd_A.copy()]
trace_mu_ikd_B = [mu_kd_B.copy()]
trace_lambda_ikdd_A = [lambda_kdd_A.copy()]
trace_lambda_ikdd_B = [lambda_kdd_B.copy()]
trace_beta_ik_A = [np.repeat(beta, K)]
trace_beta_ik_B = [np.repeat(beta, K)]
trace_m_ikd_A = [np.repeat(m_d_A.reshape((1, dim)), K, axis=0)]
trace_m_ikd_B = [np.repeat(m_d_B.reshape((1, dim)), K, axis=0)]
trace_w_ikdd_A = [np.repeat(w_dd_A.reshape((1, dim, dim)), K, axis=0)]
trace_w_ikdd_B = [np.repeat(w_dd_B.reshape((1, dim, dim)), K, axis=0)]
trace_nu_ik_A = [np.repeat(nu, K)]
trace_nu_ik_B = [np.repeat(nu, K)]

############################## M-H algorithm ##############################
iteration = args.iteration # M-H法のイテレーション
ARI_A = np.zeros((iteration))
ARI_B = np.zeros((iteration)) # 各イテレーションのARI
concidence = np.zeros((iteration)) # 各イテレーションのカッパ係数
accept_count_AtoB = np.zeros((iteration))
accept_count_BtoA = np.zeros((iteration)) # 各イテレーションの受容回数
for i in range(iteration):
    # 予測サイン
    pred_label_A = []
    pred_label_B = []
    # 現在のイテレーションでの受容回数を保存する変数
    count_AtoB = 0
    count_BtoA = 0
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~Sp:A->Li:Bここから~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    w_dk = np.zeros((D, K)); 
    for k in range(K): # Sp:A：w^Aの事後分布のパラメータを計算
        tmp_eta_nA[k] = np.diag(-0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)).copy() 
        tmp_eta_nA[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
        eta_dkA[:, k] = np.exp(tmp_eta_nA[k])
    eta_dkA /= np.sum(eta_dkA, axis=1, keepdims=True) # 正規化

    for d in range(D): # 潜在変数をサンプル：式(4.93)
        w_dk_A[d] = np.random.multinomial(n=1, pvals=eta_dkA[d], size=1).flatten() # w^Aのサンプリング
        pred_label_A.append(np.argmax(w_dk_A[d]))

        if args.mode == 0:
            # 全棄却モデル
            judge_r = -1 
        elif args.mode == 1:
            # 全受容モデル
            judge_r = 1000 
        else:
            # M-Hモデル
            # 尤度計算（緑本p65 式2.72）※式4.91じゃないよ
            cat_liks_A[d] = multivariate_normal.pdf(c_nd_B[d], 
                            mean=mu_kd_B[np.argmax(w_dk_A[d])], 
                            cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_A[d])]),
                            )
            cat_liks_B[d] = multivariate_normal.pdf(c_nd_B[d], 
                            mean=mu_kd_B[np.argmax(w_dk_B[d])], 
                            cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_B[d])]),
                            )
            judge_r = cat_liks_A[d] / cat_liks_B[d] # AとBの尤度から受容率の計算
            judge_r = min(1, judge_r) # 受容率
        rand_u = np.random.rand() # 一様変数のサンプリング
        if judge_r >= rand_u: 
            w_dk[d] = w_dk_A[d]
            count_AtoB = count_AtoB + 1 # 受容した回数をカウント
        else: 
            w_dk[d] = w_dk_B[d]

    # 更新後のw^Liを用いてエージェントBの\mu, \lambdaの再サンプリング
    for k in range(K):
        # muの事後分布のパラメータを計算
        beta_hat_k_B[k] = np.sum(w_dk[:, k]) + beta
        m_hat_kd_B[k] = np.sum(w_dk[:, k] * c_nd_B.T, axis=1)
        m_hat_kd_B[k] += beta * m_d_B; m_hat_kd_B[k] /= beta_hat_k_B[k]
        # lambdaの事後分布のパラメータを計算
        tmp_w_dd_B = np.dot((w_dk[:, k] * c_nd_B.T), c_nd_B)
        tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
        tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
        tmp_w_dd_B += np.linalg.inv(w_dd_B)
        w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
        nu_hat_k_B[k] = np.sum(w_dk[:, k]) + nu
        # 更新後のパラメータからlambdaをサンプル
        lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
        # 更新後のパラメータからmuをサンプル
        mu_kd_B[k] = np.random.multivariate_normal(mean=m_hat_kd_B[k], cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]), size=1).flatten()

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~Sp:B->Li:Aここから~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    w_dk = np.zeros((D, K)); 
    for k in range(K): # Sp:A：w^Aの事後分布のパラメータを計算
        tmp_eta_nB[k] = np.diag(-0.5 * (c_nd_B - mu_kd_B[k]).dot(lambda_kdd_B[k]).dot((c_nd_B - mu_kd_B[k]).T)).copy() 
        tmp_eta_nB[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_B[k]) + 1e-7)
        eta_dkB[:, k] = np.exp(tmp_eta_nB[k])
    eta_dkB /= np.sum(eta_dkB, axis=1, keepdims=True) # 正規化. w^Bのパラメータとなるディリクレ変数

    # 潜在変数をサンプル：式(4.93)
    for d in range(D):
        w_dk_B[d] = np.random.multinomial(n=1, pvals=eta_dkB[d], size=1).flatten() # w^Bのサンプリング
        pred_label_B.append(np.argmax(w_dk_B[d])) # 予測カテゴリ

        if args.mode == 0:
            judge_r = -1 # 全棄却用
        elif args.mode == 1:
            judge_r = 1000 # 全受容用
        else:
            cat_liks_B[d] = multivariate_normal.pdf(c_nd_A[d], 
                            mean=mu_kd_A[np.argmax(w_dk_B[d])], 
                            cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_B[d])]),
                            )
            cat_liks_A[d] = multivariate_normal.pdf(c_nd_A[d], 
                    mean=mu_kd_A[np.argmax(w_dk_A[d])], 
                    cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_A[d])]),
                    )
            judge_r = cat_liks_B[d] / cat_liks_A[d] # AとBのカテゴリ尤度から受容率の計算
            judge_r = min(1, judge_r) # 受容率
        rand_u = np.random.rand() # 一様変数のサンプリング
        if judge_r >= rand_u: 
            w_dk[d] = w_dk_B[d]
            count_BtoA = count_BtoA + 1 # 受容した回数をカウント
        else: 
            w_dk[d] = w_dk_A[d]

    # 更新後のw^Liを用いてエージェントBの\mu, \lambdaの再サンプリング
    for k in range(K):
        # muの事後分布のパラメータを計算
        beta_hat_k_A[k] = np.sum(w_dk[:, k]) + beta; m_hat_kd_A[k] = np.sum(w_dk[:, k] * c_nd_A.T, axis=1)
        m_hat_kd_A[k] += beta * m_d_A; m_hat_kd_A[k] /= beta_hat_k_A[k]
        # lambdaの事後分布のパラメータを計算
        tmp_w_dd_A = np.dot((w_dk[:, k] * c_nd_A.T), c_nd_A)
        tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
        tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
        tmp_w_dd_A += np.linalg.inv(w_dd_A)
        w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
        nu_hat_k_A[k] = np.sum(w_dk[:, k]) + nu
        # 更新後のパラメータからlambdaをサンプル
        lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
        # 更新後のパラメータからmuをサンプル
        mu_kd_A[k] = np.random.multivariate_normal(mean=m_hat_kd_A[k], cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]), size=1).flatten()

    ############################## 評価値計算 ##############################
    # cappa 係数の計算
    sum_same_w = 0.0
    a_chance = 0.0
    prob_w = [0.0 for i in range(K)]
    w_count_a = [0.0 for i in range(K)]
    w_count_b = [0.0 for i in range(K)]

    for d in range(D):
        if np.argmax(w_dk_A[d]) == np.argmax(w_dk_B[d]):
            sum_same_w += 1

        for w in range(K):
            if np.argmax(w_dk_A[d]) == w:
                w_count_a[w] += 1
            if np.argmax(w_dk_B[d]) == w:
                w_count_b[w] += 1
    
    for w in range(K):
        prob_w[w] = (w_count_a[w] / D) * (w_count_b[w] / D)
        a_chance += prob_w[w]
    a_observed = (sum_same_w / D)

    concidence[i] = np.round((a_observed - a_chance) / (1 - a_chance), 3) # Kappa係数の計算
    ARI_A[i] = np.round(ari(z_truth_n,pred_label_A),3)
    ARI_B[i] = np.round(ari(z_truth_n,pred_label_B),3) # ARI
    accept_count_AtoB[i] = count_AtoB
    accept_count_BtoA[i] = count_BtoA # 受容回数

    print(f"=> Epoch: {i+1}, ARI_A:{ARI_A[i]}, ARI_B:{ARI_B[i]}, Concidence:{concidence[i]}, Accept_AtoB:{int(accept_count_AtoB[i])}, Accept_BtoA:{int(accept_count_BtoA[i])}")
    print(f"Cappa{np.round(cohen_kappa_score(pred_label_A,pred_label_B),3)}")
    # 値を記録
    _, w_n_A = np.where(w_dk_A == 1)
    _, w_n_B = np.where(w_dk_B == 1)
    trace_w_in_A.append(w_n_A.copy())
    trace_w_in_B.append(w_n_B.copy())
    trace_mu_ikd_A.append(mu_kd_A.copy())
    trace_mu_ikd_B.append(mu_kd_B.copy())
    trace_lambda_ikdd_A.append(lambda_kdd_A.copy())
    trace_lambda_ikdd_B.append(lambda_kdd_B.copy())
    trace_beta_ik_A.append(beta_hat_k_A.copy())
    trace_beta_ik_B.append(beta_hat_k_B.copy())
    trace_m_ikd_A.append(m_hat_kd_A.copy())
    trace_m_ikd_B.append(m_hat_kd_B.copy())
    trace_w_ikdd_A.append(w_hat_kdd_A.copy())
    trace_w_ikdd_B.append(w_hat_kdd_B.copy())
    trace_nu_ik_A.append(nu_hat_k_A.copy())
    trace_nu_ik_B.append(nu_hat_k_B.copy())

# 受容回数
plt.figure()
#plt.ylim(0,)
plt.plot(range(0,iteration), accept_count_AtoB, marker="None", label="Accept_num:AtoB")
plt.plot(range(0,iteration), accept_count_BtoA, marker="None", label="Accept_num:BtoA")
plt.xlabel('iteration')
plt.ylabel('Number of acceptation')
plt.ylim(0,D)
plt.legend()
plt.savefig(dir_name+'/accept.png')
#plt.show()
plt.close()

# concidence
plt.plot(range(0,iteration), concidence, marker="None")
plt.xlabel('iteration')
plt.ylabel('Concidence')
plt.ylim(0,1)
plt.title('Cappa')
plt.savefig(dir_name+"/conf.png")
#plt.show()
plt.close()

# ARI
plt.plot(range(0,iteration), ARI_A, marker="None",label="ARI_A")
plt.plot(range(0,iteration), ARI_B, marker="None",label="ARI_B")
plt.xlabel('iteration')
plt.ylabel('ARI')
plt.ylim(0,1)
plt.legend()
plt.title('ARI')
plt.savefig(dir_name+"/ari.png")
#plt.show()
plt.close()

# 受容率
plt.plot(range(0,D), cat_liks_A, marker="None",label="cat_liks_A")
plt.plot(range(0,D), cat_liks_B, marker="None",label="cat_liks_B")
plt.xlabel('Data num')
plt.ylabel('liks')
plt.ylim(0,1)
plt.legend()
plt.title('liks')
plt.savefig(dir_name+"/liks.png")
#plt.show()
plt.close()