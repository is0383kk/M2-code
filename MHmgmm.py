import os
import numpy as np
from scipy.stats import wishart, multivariate_normal
import matplotlib.pyplot as plt
from tool import calc_ari
from sklearn.metrics import cohen_kappa_score
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import argparse
from tool import visualize_gmm
from sklearn.metrics.cluster import adjusted_rand_score as ari
import cnn_vae_module
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def print_cmx(y_true, y_pred, agent):
    labels = sorted(list(set(y_true)))
    cmx = confusion_matrix(y_true, y_pred, labels=labels)
    #cmd = ConfusionMatrixDisplay(cmx,display_labels=None)
    #cmd.plot()
    df_cmx = pd.DataFrame(cmx, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=False)
    plt.savefig("cm_"+agent+".png")
    #plt.show()

parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM Example')

parser.add_argument('--mode', type=int, default=-1, metavar='M', help='0:All reject, 1:ALL accept')
parser.add_argument('--iteration', type=int, default=100, metavar='N', help='number of iteration for MGMM_MH')
args = parser.parse_args()

############################## Prepareing Dataset ##############################
# MNIST左右回転設定
angle_a = 0 # 回転角度
angle_b = 75 # 回転角度
trans_ang1 = transforms.Compose([transforms.RandomRotation(degrees=(angle_a, angle_a)), transforms.ToTensor()]) # -angle度回転設定
trans_ang2 = transforms.Compose([transforms.RandomRotation(degrees=(angle_b, angle_b)), transforms.ToTensor()]) # angle度回転設定
# データセット定義
trainval_dataset1 = datasets.MNIST('./../data', train=True, transform=trans_ang1, download=False) # Agent A用 MNIST
trainval_dataset2 = datasets.MNIST('./../data', train=True, transform=trans_ang2, download=False) # Agent B用 MNIST
n_samples = len(trainval_dataset1)
D = int(n_samples * (1/6)) # データ総数
subset1_indices1 = list(range(0, D)); subset2_indices1 = list(range(D, n_samples)) 
subset1_indices2 = list(range(0, D)); subset2_indices2 = list(range(D, n_samples)) 
train_dataset1 = Subset(trainval_dataset1, subset1_indices1); val_dataset1 = Subset(trainval_dataset1, subset2_indices1)
train_dataset2 = Subset(trainval_dataset2, subset1_indices1); val_dataset2 = Subset(trainval_dataset2, subset2_indices2)
all_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=D, shuffle=False) # データセット総数分のローダ
all_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=D, shuffle=False) # データセット総数分のローダ



# 各種保存用ディレクトリの作成
file_name = "MNISTa0b75"; model_dir = "./MNISTa0b75"; dir_name = "./model/"+file_name# debugフォルダに保存される
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)

K = 10
c_nd_A, z_truth_n = cnn_vae_module.send_all_z(iteration=2, all_loader=all_loader1, model_dir=dir_name, agent="A")
c_nd_B, z_truth_n = cnn_vae_module.send_all_z(iteration=2, all_loader=all_loader2, model_dir=dir_name, agent="B")

D = len(c_nd_A)
dim = len(c_nd_A[0])
print(f"Number of clusters: {K}"); print(f"Number of data: {len(c_nd_A)}"); print(f"Number of dimention: {len(c_nd_A[0])}")

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
    mu_kd_A[k] = np.random.multivariate_normal(mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])).flatten()
    mu_kd_B[k] = np.random.multivariate_normal(mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])).flatten()

# Initializing unsampled \w
w_dk_A = np.random.multinomial(1, [1/K]*K, size=D); w_dk_B = np.random.multinomial(1, [1/K]*K, size=D)

# 各種パラメータの初期化
beta_hat_k_A = np.zeros(K) ;beta_hat_k_B = np.zeros(K)
m_hat_kd_A = np.zeros((K, dim)); m_hat_kd_B = np.zeros((K, dim))
w_hat_kdd_A = np.zeros((K, dim, dim)); w_hat_kdd_B = np.zeros((K, dim, dim))
nu_hat_k_A = np.zeros(K); nu_hat_k_B = np.zeros(K)
tmp_eta_nB = np.zeros((K, D)); eta_dkB = np.zeros((D, K))
tmp_eta_nA = np.zeros((K, D)); eta_dkA = np.zeros((D, K))
cat_liks_A = np.zeros(D); cat_liks_B = np.zeros(D)
mu_d_A = np.zeros((D,dim)); var_d_A = np.zeros((D,dim))
mu_d_B = np.zeros((D,dim)); var_d_B = np.zeros((D,dim))

# 推移保存用
trace_w_in_A = [np.repeat(np.nan, D)]; trace_w_in_B = [np.repeat(np.nan, D)]
trace_mu_ikd_A = [mu_kd_A.copy()]; trace_mu_ikd_B = [mu_kd_B.copy()]
trace_lambda_ikdd_A = [lambda_kdd_A.copy()]; trace_lambda_ikdd_B = [lambda_kdd_B.copy()]
trace_beta_ik_A = [np.repeat(beta, K)]; trace_beta_ik_B = [np.repeat(beta, K)]
trace_m_ikd_A = [np.repeat(m_d_A.reshape((1, dim)), K, axis=0)]; trace_m_ikd_B = [np.repeat(m_d_B.reshape((1, dim)), K, axis=0)]
trace_w_ikdd_A = [np.repeat(w_dd_A.reshape((1, dim, dim)), K, axis=0)]; trace_w_ikdd_B = [np.repeat(w_dd_B.reshape((1, dim, dim)), K, axis=0)]
trace_nu_ik_A = [np.repeat(nu, K)]; trace_nu_ik_B = [np.repeat(nu, K)]

iteration = 100
ARI_A = np.zeros((iteration)); ARI_B = np.zeros((iteration)); 
concidence = np.zeros((iteration))
accept_count_AtoB = np.zeros((iteration)); accept_count_BtoA = np.zeros((iteration)) # Number of acceptation
############################## M-H algorithm ##############################
print("M-H algorithm")
for i in range(iteration):
    pred_label_A = []; pred_label_B = []
    count_AtoB = count_BtoA = 0 # 現在のイテレーションでの受容回数を保存する変数
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~Sp:A->Li:Bここから~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    w_dk = np.zeros((D, K)); 
    for k in range(K): # Sp:A：w^Aの事後分布のパラメータを計算
        tmp_eta_nA[k] = np.diag(-0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)).copy() 
        #print(f"tmp_eta_nA:{np.round(tmp_eta_nA[k],4)}")
        tmp_eta_nA[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
        eta_dkA[:, k] = np.exp(tmp_eta_nA[k])
    eta_dkA /= np.sum(eta_dkA, axis=1, keepdims=True) # 正規化.w^Aのパラメータとなるディリクレ変数

    for d in range(D): # 潜在変数をサンプル：式(4.93)
        w_dk_A[d] = np.random.multinomial(n=1, pvals=eta_dkA[d], size=1).flatten() # w^Aのサンプリング
        #pred_label_A.append(np.argmax(w_dk_A[d]))
        
        if args.mode == 0:
            judge_r = -1 # 全棄却用
        elif args.mode == 1:
            judge_r = 1000 # 全棄却用
        else:
            cat_liks_A[d] = multivariate_normal.pdf(c_nd_B[d], 
                            mean=mu_kd_B[np.argmax(w_dk_A[d])], 
                            cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_A[d])]),
                            )
            cat_liks_B[d] = multivariate_normal.pdf(c_nd_B[d], 
                            mean=mu_kd_B[np.argmax(w_dk_B[d])], 
                            cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_B[d])]),
                            )
            judge_r = cat_liks_A[d] / cat_liks_B[d] # AとBのカテゴリ尤度から受容率の計算
            judge_r = min(1, judge_r) # 受容率
        rand_u = np.random.rand() # 一様変数のサンプリング
        if judge_r >= rand_u:
            w_dk[d] = w_dk_A[d]
            count_AtoB = count_AtoB + 1 # 受容した回数をカウント
        else: 
            w_dk[d] = w_dk_B[d]
        pred_label_B.append(np.argmax(w_dk[d])) # 予測カテゴリ

    # 更新後のw^Liを用いてエージェントBの\mu, \lambdaの再サンプリング
    for k in range(K):
        # muの事後分布のパラメータを計算
        beta_hat_k_B[k] = np.sum(w_dk[:, k]) + beta; m_hat_kd_B[k] = np.sum(w_dk[:, k] * c_nd_B.T, axis=1)
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

        """
        for d in range(D):
            mu_d_B[d] = mu_kd_B[np.argmax(w_dk[d])]
            var_d_B[d] = np.diag(np.linalg.inv(lambda_kdd_B[np.argmax(w_dk[d])]))
        """

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
        #pred_label_B.append(np.argmax(w_dk_B[d])) # 予測カテゴリ
        
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
        pred_label_A.append(np.argmax(w_dk[d])) # 予測カテゴリ
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
        """
        for d in range(D):
            mu_d_A[d] = mu_kd_A[np.argmax(w_dk[d])]
            var_d_A[d] = np.diag(np.linalg.inv(lambda_kdd_A[np.argmax(w_dk[d])]))
        """

    ############################## 評価値計算 ##############################
    # Kappa係数の計算
    concidence[i] = np.round(cohen_kappa_score(pred_label_A,pred_label_B),3)
    # ARIの計算
    ARI_A[i], result_a = calc_ari(pred_label_A, z_truth_n) 
    ARI_B[i], result_b = calc_ari(pred_label_B, z_truth_n)
    # 受容回数
    accept_count_AtoB[i] = count_AtoB; accept_count_BtoA[i] = count_BtoA
    
    if i == 0 or (i+1) % 10 == 0 or i == (iteration-1): 
        print(f"=> Ep: {i+1}, A: {ARI_A[i]}, B: {ARI_B[i]}, C:{concidence[i]}, A2B:{int(accept_count_AtoB[i])}, B2A:{int(accept_count_BtoA[i])}")
    for d in range(D):
        mu_d_A[d] = mu_kd_A[np.argmax(w_dk[d])]
        var_d_A[d] = np.diag(np.linalg.inv(lambda_kdd_A[np.argmax(w_dk[d])]))
        mu_d_B[d] = mu_kd_B[np.argmax(w_dk[d])]
        var_d_B[d] = np.diag(np.linalg.inv(lambda_kdd_B[np.argmax(w_dk[d])]))
    
    # 値を記録
    _, w_n_A = np.where(w_dk_A == 1); _, w_n_B = np.where(w_dk_B == 1)
    trace_w_in_A.append(w_n_A.copy()); trace_w_in_B.append(w_n_B.copy())
    trace_mu_ikd_A.append(mu_kd_A.copy()); trace_mu_ikd_B.append(mu_kd_B.copy())
    trace_lambda_ikdd_A.append(lambda_kdd_A.copy()); trace_lambda_ikdd_B.append(lambda_kdd_B.copy())
    trace_beta_ik_A.append(beta_hat_k_A.copy()); trace_beta_ik_B.append(beta_hat_k_B.copy())
    trace_m_ikd_A.append(m_hat_kd_A.copy()); trace_m_ikd_B.append(m_hat_kd_B.copy())
    trace_w_ikdd_A.append(w_hat_kdd_A.copy()); trace_w_ikdd_B.append(w_hat_kdd_B.copy())
    trace_nu_ik_A.append(nu_hat_k_A.copy()); trace_nu_ik_B.append(nu_hat_k_B.copy())

print_cmx(z_truth_n, result_a, agent="A")
print_cmx(z_truth_n, result_b, agent="B")    
print(f"maxA:{max(ARI_A)}, max_B:{max(ARI_B)}, max_c:{max(concidence)}")

