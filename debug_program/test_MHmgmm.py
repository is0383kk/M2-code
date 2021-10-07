import os
import numpy as np
from scipy.stats import wishart 
import matplotlib.pyplot as plt
from tool import calc_ari
from sklearn.metrics import cohen_kappa_score
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import argparse
#from custom_data import CustomDataset
from tool import visualize_gmm

parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='input batch size for training')
parser.add_argument('--iteration', type=int, default=10, metavar='N', help='number of learning iteration')
parser.add_argument('--category', type=int, default=10, metavar='N', help='number of category for GMM module')
parser.add_argument('--rate', type=int, default=100, metavar='N', help='number of category for GMM module')
#parser.add_argument('--iteration', type=int, default=100, metavar='N', help='number of iteration for MGMM_MH')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

# 各種保存用ディレクトリの作成
file_name = "debug"; model_dir = "./model"; dir_name = "./model/"+file_name# debugフォルダに保存される
graphA_dir = "./model/"+file_name+"/graphA"; graphB_dir = "./model/"+file_name+"/graphB" # 各種グラフの保存先
pthA_dir = "./model/"+file_name+"/pthA"; pthB_dir = "./model/"+file_name+"/pthB"; npy_dir = "./model/"+file_name+"/npy"
reconA_dir = model_dir+"/"+file_name+"/reconA"; reconB_dir = model_dir+"/"+file_name+"/reconB"
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)
if not os.path.exists(pthA_dir):    os.mkdir(pthA_dir)
if not os.path.exists(pthB_dir):    os.mkdir(pthB_dir)
if not os.path.exists(graphA_dir):   os.mkdir(graphA_dir)
if not os.path.exists(graphB_dir):   os.mkdir(graphB_dir)
if not os.path.exists(npy_dir):    os.mkdir(npy_dir)
if not os.path.exists(reconA_dir):    os.mkdir(reconA_dir)
if not os.path.exists(reconB_dir):    os.mkdir(reconB_dir)


K = 4
c_nd_A = np.loadtxt("./dataset/data1.txt");c_nd_B = np.loadtxt("./dataset/data2.txt");z_truth_n = np.loadtxt("./dataset/true_label.txt") 
#c_nd_A = np.loadtxt("./samedata.txt") c_nd_B = np.loadtxt("./samedata.txt");z_truth_n = np.loadtxt("./samelabel.txt")
D = len(c_nd_A)
dim = len(c_nd_A[0])
print(f"Number of clusters: {K}"); print(f"Number of data: {len(c_nd_A)}"); print(f"Number of dimention: {len(c_nd_A[0])}")

iteration = 1
ARI_A = np.zeros((iteration)); ARI_B = np.zeros((iteration)); max_A_ARI = 0; max_B_ARI = 0
concidence = np.zeros((iteration))
accept_count_AtoB = np.zeros((iteration)); accept_count_BtoA = np.zeros((iteration)) # Number of acceptation

############################## Initializing parameters ##############################
print("Initializing parameters")
# Set hyperparameters
beta = 1.0; m_d_A = np.repeat(0.0, dim); m_d_B = np.repeat(0.0, dim) # Hyperparameters for \mu^A, \mu^B
w_dd_A = np.identity(dim) * 0.05; w_dd_B = np.identity(dim) * 0.05 # Hyperparameters for \Lambda^A, \Lambda^B
nu = dim

# Initializing \mu, \Lambda
mu_kd_A = np.empty((K, dim)); lambda_kdd_A = np.empty((K, dim, dim))
mu_kd_B = np.empty((K, dim)); lambda_kdd_B = np.empty((K, dim, dim))
for k in range(K):
    lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1); lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
    mu_kd_A[k] = np.random.multivariate_normal(
        mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])
    ).flatten()
    mu_kd_B[k] = np.random.multivariate_normal(
        mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])
    ).flatten()

# Initializing unsampled \w

rand_set_a = []
rand_set_b = []
for d in range(D):
    rand_a = np.random.randint(0, K)
    rand_set_a.append(rand_a)
    rand_b = np.random.randint(0, K)
    rand_set_b.append(rand_b)
print(rand_set_a)
#w_dk_A = np.zeros((D, K)); w_dk_B = np.zeros((D, K))
w_dk_A = np.random.multinomial(1, [1/K]*K, size=D); w_dk_B = np.random.multinomial(1, [1/K]*K, size=D)
#print(f"w_dk_A:{np.random.multinomial(1, [1/K]*K, size=D)}")

# Initializing learning parameters
beta_hat_k_A = np.zeros(K) ;beta_hat_k_B = np.zeros(K)
m_hat_kd_A = np.zeros((K, dim)); m_hat_kd_B = np.zeros((K, dim))
w_hat_kdd_A = np.zeros((K, dim, dim)); w_hat_kdd_B = np.zeros((K, dim, dim))
nu_hat_k_A = np.zeros(K); nu_hat_k_B = np.zeros(K)

# Variables for storing the transition of each parameter
trace_w_in_A = [np.repeat(np.nan, D)]; trace_w_in_B = [np.repeat(np.nan, D)]
trace_mu_ikd_A = [mu_kd_A.copy()]; trace_mu_ikd_B = [mu_kd_B.copy()]
trace_lambda_ikdd_A = [lambda_kdd_A.copy()]; trace_lambda_ikdd_B = [lambda_kdd_B.copy()]
trace_beta_ik_A = [np.repeat(beta, K)]; trace_beta_ik_B = [np.repeat(beta, K)]
trace_m_ikd_A = [np.repeat(m_d_A.reshape((1, dim)), K, axis=0)]; trace_m_ikd_B = [np.repeat(m_d_B.reshape((1, dim)), K, axis=0)]
trace_w_ikdd_A = [np.repeat(w_dd_A.reshape((1, dim, dim)), K, axis=0)]; trace_w_ikdd_B = [np.repeat(w_dd_B.reshape((1, dim, dim)), K, axis=0)]
trace_nu_ik_A = [np.repeat(nu, K)]; trace_nu_ik_B = [np.repeat(nu, K)]

############################## M-H algorithm ##############################
print("M-H algorithm")
for i in range(iteration):
    # Initializing z
    print(f"-------------{i+1}試行目-------------")
    count_AtoB = 0
    count_BtoA = 0
    #########################################################################A->Bここから
    pred_label_A = []
    pred_label_B = []
    # wのパラメータを計算
    tmp_eta_nA = np.zeros((K, D)); tmp_eta_nB = np.zeros((K, D))
    eta_dkA = np.zeros((D, K)); eta_dkB = np.zeros((D, K))
    for k in range(K):
        # エージェントA：更新後の\mu^A,\Lambda^Aからw^Aの事後分布のパラメータを計算
        tmp_eta_nA[k] = np.diag(
            -0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)
        ).copy() 
        tmp_eta_nA[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
        eta_dkA[:, k] = np.exp(tmp_eta_nA[k])
        # エージェントB：１イテレーション前の\mu^B,\Lambda^Bからw^Bの事後分布のパラメータを計算
        tmp_eta_nB[k] = np.diag(
            -0.5 * (c_nd_B - mu_kd_B[k]).dot(lambda_kdd_B[k]).dot((c_nd_B - mu_kd_B[k]).T)
        ).copy() 
        tmp_eta_nB[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_B[k]) + 1e-7)
        eta_dkB[:, k] = np.exp(tmp_eta_nB[k])
    print(f"tmp_eta_nB:{np.round(tmp_eta_nB,3)}")
    #print(f"eta_dkB:{np.round(eta_dkB,3)}")
    eta_dkA /= np.sum(eta_dkA, axis=1, keepdims=True) # Normalization
    eta_dkB /= np.sum(eta_dkB, axis=1, keepdims=True) # Normalization
    #print(f"eta_dkB:{np.round(eta_dkB,3)}")
    # 潜在変数をサンプル：式(4.93)
    for d in range(D):
        #print(f"-------------潜在変数割当て：D={d}-------------")
        w_dk_A[d] = np.random.multinomial(n=1, pvals=eta_dkA[d], size=1).flatten() # w^Aのサンプリング
        w_dk_B[d] = np.random.multinomial(n=1, pvals=eta_dkB[d], size=1).flatten() # w^Bのカテゴリ尤度計算用にサンプリング
        pred_label_A.append(np.argmax(w_dk_A[d])) 
        #print(f"w_dk_A[d]:{w_dk_A[d]},インデックス:{np.argmax(w_dk_A[d])}")
        #print(f"eta_dkA:{eta_dkA[d]},{eta_dkA[d][np.argmax(w_dk_A[d])]}")
        #print(f"{eta_dkA[d][0]},{eta_dkA[d][1]},{eta_dkA[d][2]}")

        # 尤度比較
        #cat_liks_A = tmp_eta_nA.T[d][np.where(w_dk_A[d]==1)]
        #cat_liks_B = tmp_eta_nB.T[d][np.where(w_dk_B[d]==1)]
        #judge_r = cat_liks_B / cat_liks_A # AとBのカテゴリ尤度から受容率の計算
        
        # ディリクレ比較
        cat_liks_A = eta_dkA[d][np.argmax(w_dk_A[d])]
        cat_liks_B = eta_dkB[d][np.argmax(w_dk_B[d])]
        judge_r = cat_liks_A / cat_liks_B # AとBのカテゴリ尤度から受容率の計算
        
        rand_u = np.random.rand() # 一様変数のサンプリング
        #print(f"rate={np.round(judge_r,3)}:c_liks1={np.round(cat_liks_A,3)}, c_liks2={np.round(cat_liks_B,3)}, u={np.round(rand_u,3)}") 
        judge_r = min(1, judge_r) # 受容率
        #judge_r = -1 # 受容率
        #judge_r = 1000 # 受容率
        if judge_r >= rand_u: 
            # 受容
            w_dk_B[d] = w_dk_A[d] # w_d = w_d^{Sp}
            count_AtoB = count_AtoB + 1 # 受容した回数をカウント

    # 更新後のサインを用いてエージェントBの\mu, \lambdaの再サンプリング
    # 観測モデルのパラメータをサンプル
    for k in range(K):
        # muの事後分布のパラメータを計算
        beta_hat_k_B[k] = np.sum(w_dk_B[:, k]) + beta; m_hat_kd_B[k] = np.sum(w_dk_B[:, k] * c_nd_B.T, axis=1)
        m_hat_kd_B[k] += beta * m_d_B; m_hat_kd_B[k] /= beta_hat_k_B[k]
        # lambdaの事後分布のパラメータを計算
        tmp_w_dd_B = np.dot((w_dk_B[:, k] * c_nd_B.T), c_nd_B)
        tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
        tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
        tmp_w_dd_B += np.linalg.inv(w_dd_B)
        w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
        nu_hat_k_B[k] = np.sum(w_dk_B[:, k]) + nu
        # 更新後のパラメータからlambdaをサンプル
        lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
        # 更新後のパラメータからmuをサンプル
        mu_kd_B[k] = np.random.multivariate_normal(
            mean=m_hat_kd_B[k], cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]), size=1
        ).flatten()
    #########################################################################A->Bここまで
    
    mu_d_B = np.zeros((D,dim)) # GMMの平均パラメータ
    var_d_B = np.zeros((D,dim)) # GMMのLambdaの対角成分（分散）
    for d in range(D):
        pred_label_B.append(np.argmax(w_dk_B[d]))
        var_d_B[d] = np.diag(np.linalg.inv(lambda_kdd_B[pred_label_B[d]]))
        mu_d_B[d] = mu_d_B[pred_label_B[d]]


    #########################################################################B->Aここから
    
    pred_label_B = []
    pred_label_A = []
    # wのパラメータを計算
    tmp_eta_nA = np.zeros((K, D)); tmp_eta_nB = np.zeros((K, D))
    eta_dkA = np.zeros((D, K)); eta_dkB = np.zeros((D, K))
    for k in range(K):
        #print(f"-------------カテゴリ尤度計算：K={k}-------------")
        # エージェントA：更新後の\mu,\Lambdaからw^Aの事後分布のパラメータを計算
        tmp_eta_nB[k] = np.diag(
            -0.5 * (c_nd_B - mu_kd_B[k]).dot(lambda_kdd_B[k]).dot((c_nd_B - mu_kd_B[k]).T)
        ).copy() 
        tmp_eta_nB[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_B[k]) + 1e-7)
        eta_dkB[:, k] = np.exp(tmp_eta_nB[k])
        # エージェントB：１イテレーション前の\mu,\Lambdaからw^Bの事後分布のパラメータを計算
        tmp_eta_nA[k] = np.diag(
            -0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)
        ).copy() 
        tmp_eta_nA[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
        eta_dkA[:, k] = np.exp(tmp_eta_nA[k])

    eta_dkB /= np.sum(eta_dkB, axis=1, keepdims=True) # 正規化. w^Aのパラメータとなるディリクレ変数
    eta_dkA /= np.sum(eta_dkA, axis=1, keepdims=True) # 正規化. w^Bのパラメータとなるディリクレ変数

    # 潜在変数をサンプル：式(4.93)
    for d in range(D):
        #print(f"-------------潜在変数割当て：D={d}-------------")
        w_dk_B[d] = np.random.multinomial(n=1, pvals=eta_dkB[d], size=1).flatten() # w^Bのサンプリング
        w_dk_A[d] = np.random.multinomial(n=1, pvals=eta_dkA[d], size=1).flatten() # w^Aのカテゴリ尤度計算用にサンプリング
        pred_label_B.append(np.argmax(w_dk_B[d]))
        
        # 尤度比較
        #cat_liks_B = tmp_eta_nB.T[d][np.where(w_dk_B[d]==1)]
        #cat_liks_A = tmp_eta_nA.T[d][np.where(w_dk_A[d]==1)]
        #judge_r = cat_liks_A / cat_liks_B # AとBのカテゴリ尤度から受容率の計算
        
        # ディリクレ比較
        cat_liks_B = eta_dkB[d][np.argmax(w_dk_B[d])]
        cat_liks_A = eta_dkA[d][np.argmax(w_dk_A[d])]
        judge_r = cat_liks_B / cat_liks_A # AとBのカテゴリ尤度から受容率の計算
        
        rand_u = np.random.rand() # 一様変数のサンプリング
        judge_r = min(1, judge_r) # 受容率
        judge_r = -1 # 受容率
        judge_r = 1000 # 受容率
        if judge_r >= rand_u: 
            # 受容
            w_dk_A[d] = w_dk_B[d] # w_d = w_d^{Sp}
            count_BtoA = count_BtoA + 1 # 受容した回数をカウント
        
        
    # 更新後のサインを用いてエージェントBの\mu, \lambdaの再サンプリング
    # 観測モデルのパラメータをサンプル
    for k in range(K):
        # muの事後分布のパラメータを計算
        beta_hat_k_A[k] = np.sum(w_dk_A[:, k]) + beta; m_hat_kd_A[k] = np.sum(w_dk_A[:, k] * c_nd_A.T, axis=1)
        m_hat_kd_A[k] += beta * m_d_A; m_hat_kd_A[k] /= beta_hat_k_A[k]
        # lambdaの事後分布のパラメータを計算
        tmp_w_dd_A = np.dot((w_dk_A[:, k] * c_nd_A.T), c_nd_A)
        tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
        tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
        tmp_w_dd_A += np.linalg.inv(w_dd_A)
        w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
        nu_hat_k_A[k] = np.sum(w_dk_A[:, k]) + nu
        # 更新後のパラメータからlambdaをサンプル
        lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
        # 更新後のパラメータからmuをサンプル
        mu_kd_A[k] = np.random.multivariate_normal(
            mean=m_hat_kd_A[k], cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]), size=1
        ).flatten()
    #########################################################################B->Aここまで
    mu_d_A = np.zeros((D,dim)) # GMMの平均パラメータ
    var_d_A = np.zeros((D,dim)) # GMMのLambdaの対角成分（分散）
    for d in range(D):
        pred_label_A.append(np.argmax(w_dk_A[d]))
        var_d_A[d] = np.diag(np.linalg.inv(lambda_kdd_A[pred_label_A[d]]))
        mu_d_A[d] = mu_d_A[pred_label_A[d]]
    
    
    ############################## 評価値計算 ##############################
    # cappa 係数の計算
    sum_same_w = 0.0
    a_chance = 0.0
    prob_w = [0.0 for i in range(K)]
    w_count_a = [0.0 for i in range(K)]
    w_count_b = [0.0 for i in range(K)]

    """
    for d in range(D):
        #print(f"rand_set_a: {rand_set_a},{len(rand_set_a)}")
        if rand_set_a[d] == rand_set_b[d]:
            sum_same_w += 1

        for w in range(K):
            if rand_set_a[d] == w:
                w_count_a[w] += 1
            if rand_set_b[d] == w:
                w_count_b[w] += 1
    """
    
    for d in range(D):
        #if pred_label_A[d] == pred_label_B[d]:
        #print(f"pred_a:{pred_label_A[d]}")
        #print(f"w_dk_A:{np.argmax(w_dk_B[d])}")
        if np.argmax(w_dk_A[d]) == np.argmax(w_dk_B[d]):
            #print(f"np.argmax(w_dk_A[d]):{np.argmax(w_dk_A[d])}")
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

    # Kappa係数の計算
    concidence[i] = np.round((a_observed - a_chance) / (1 - a_chance), 3)

    #ARI[i] = np.round(calc_ari(pred_label_A,label)[0],4); print(f"ARI:{ARI[i]}")
    ARI_A[i] = np.round(calc_ari(pred_label_A, z_truth_n)[0],3); ARI_B[i] = np.round(calc_ari(pred_label_B, z_truth_n)[0],3)
    accept_count_AtoB[i] = count_AtoB; accept_count_BtoA[i] = count_BtoA
    
    if max_A_ARI <= ARI_A[i]: max_A_ARI = ARI_A[i]        
    if i == 0 or (i+1) % 50 == 0 or i == (iteration-1): print(f"====> Epoch: {i+1}, ARI_A: {ARI_A[i]}, ARI_B: {ARI_B[i]} Max_A_ARI: {max_A_ARI}")
    print(f"ARI_A:{ARI_A[i]}, ARI_B:{ARI_B[i]}, cappa:{concidence[i]}")
    #print(f"Accept_count(A->B):{accept_count_AtoB}"); print(f"Accept_count(B->A):{accept_count_BtoA}")

    # 値を記録
    _, w_n_A = np.where(w_dk_A == 1); _, w_n_B = np.where(w_dk_B == 1)
    trace_w_in_A.append(w_n_A.copy()); trace_w_in_B.append(w_n_B.copy())
    trace_mu_ikd_A.append(mu_kd_A.copy()); trace_mu_ikd_B.append(mu_kd_B.copy())
    trace_lambda_ikdd_A.append(lambda_kdd_A.copy()); trace_lambda_ikdd_B.append(lambda_kdd_B.copy())
    trace_beta_ik_A.append(beta_hat_k_A.copy()); trace_beta_ik_B.append(beta_hat_k_B.copy())
    trace_m_ikd_A.append(m_hat_kd_A.copy()); trace_m_ikd_B.append(m_hat_kd_B.copy())
    trace_w_ikdd_A.append(w_hat_kdd_A.copy()); trace_w_ikdd_B.append(w_hat_kdd_B.copy())
    trace_nu_ik_A.append(nu_hat_k_A.copy()); trace_nu_ik_B.append(nu_hat_k_B.copy())
    np.save(npy_dir+'/muA_'+str(iteration)+'.npy', mu_kd_A); np.save(npy_dir+'/muB_'+str(iteration)+'.npy', mu_kd_B)
    np.save(npy_dir+'/lambdaA_'+str(iteration)+'.npy', lambda_kdd_A); np.save(npy_dir+'/lambdaB_'+str(iteration)+'.npy', lambda_kdd_B)


# 受容回数
plt.figure()
#plt.ylim(0,)
plt.plot(range(0,iteration), accept_count_AtoB, marker="None", label="Accept_num:AtoB")
plt.plot(range(0,iteration), accept_count_BtoA, marker="None", label="Accept_num:BtoA")
plt.xlabel('iteration');plt.ylabel('Number of acceptation')
plt.ylim(0,D)
plt.legend()
plt.savefig(dir_name+'/accept.png')
#plt.show()
plt.close()

# concidence
plt.plot(range(0,iteration), concidence, marker="None")
plt.xlabel('iteration'); plt.ylabel('Concidence')
plt.title('Cappa')
plt.savefig(dir_name+"/conf.png")
#plt.show()
plt.close()

# ARI
plt.plot(range(0,iteration), ARI_A, marker="None",label="ARI_A")
plt.plot(range(0,iteration), ARI_B, marker="None",label="ARI_B")
plt.xlabel('iteration'); plt.ylabel('ARI')
plt.ylim(0,)
plt.legend()
plt.title('ARI')
plt.savefig(dir_name+"/ari.png")
#plt.show()
plt.close()