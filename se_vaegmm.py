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
#from custom_data import CustomDataset
from tool import visualize_gmm

parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='B', help='input batch size for training')
parser.add_argument('--vae-iter', type=int, default=100, metavar='V', help='number of VAE iteration')
parser.add_argument('--mh-iter', type=int, default=100, metavar='M', help='number of M-H mgmm iteration')
parser.add_argument('--category', type=int, default=10, metavar='K', help='number of category for GMM module')
parser.add_argument('--mode', type=int, default=-1, metavar='M', help='0:All reject, 1:ALL accept')
parser.add_argument('--debug', type=bool, default=False, metavar='D', help='Debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
if args.debug is True: args.vae_iter=2; args.mh_iter=2

############################## Making directory ##############################
file_name = "debug"; model_dir = "./model"; dir_name = "./model/"+file_name# debugフォルダに保存される
graphA_dir = "./model/"+file_name+"/graphA"; graphB_dir = "./model/"+file_name+"/graphB" # 各種グラフの保存先
pth_dir = "./model/"+file_name+"/pth";npy_dir = "./model/"+file_name+"/npy"
reconA_dir = model_dir+"/"+file_name+"/reconA"; reconB_dir = model_dir+"/"+file_name+"/reconB"
log_dir = model_dir+"/"+file_name+"/log"; result_dir = model_dir+"/"+file_name+"/result"
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)
if not os.path.exists(pth_dir):    os.mkdir(pth_dir)
if not os.path.exists(graphA_dir):   os.mkdir(graphA_dir)
if not os.path.exists(graphB_dir):   os.mkdir(graphB_dir)
if not os.path.exists(npy_dir):    os.mkdir(npy_dir)
if not os.path.exists(reconA_dir):    os.mkdir(reconA_dir)
if not os.path.exists(reconB_dir):    os.mkdir(reconB_dir)
if not os.path.exists(log_dir):    os.mkdir(log_dir)
if not os.path.exists(result_dir):    os.mkdir(result_dir)

############################## Prepareing Dataset ##############################
# MNIST左右回転設定
angle = 90 # 回転角度
trans_ang1 = transforms.Compose([transforms.RandomRotation(degrees=(-angle,-angle)), transforms.ToTensor()]) # -angle度回転設定
trans_ang2 = transforms.Compose([transforms.RandomRotation(degrees=(angle,angle)), transforms.ToTensor()]) # angle度回転設定
# データセット定義
trainval_dataset1 = datasets.MNIST('./../data', train=True, transform=trans_ang1, download=False) # Agent A用 MNIST
trainval_dataset2 = datasets.MNIST('./../data', train=True, transform=trans_ang2, download=False) # Agent B用 MNIST
n_samples = len(trainval_dataset1)
D = int(n_samples * (1/6)) # データ総数
subset1_indices1 = list(range(0, D)); subset2_indices1 = list(range(D, n_samples)) 
subset1_indices2 = list(range(0, D)); subset2_indices2 = list(range(D, n_samples)) 
train_dataset1 = Subset(trainval_dataset1, subset1_indices1); val_dataset1 = Subset(trainval_dataset1, subset2_indices1)
train_dataset2 = Subset(trainval_dataset2, subset1_indices1); val_dataset2 = Subset(trainval_dataset2, subset2_indices2)
train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=False) # train_loader for agent A
train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=False) # train_loader for agent B
all_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=D, shuffle=False) # データセット総数分のローダ
all_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=D, shuffle=False) # データセット総数分のローダ
print(f"D={D}, VAE_iter:{args.vae_iter}, MH_iter:{args.mh_iter}, MH_mode:{args.mode}"); 

import vae_module

mutual_iteration = 5
mu_d_A = np.zeros((D)); var_d_A = np.zeros((D)) 
mu_d_B = np.zeros((D)); var_d_B = np.zeros((D))
for it in range(mutual_iteration):
    print(f"------------------Mutual learning session {it} begins------------------")
    ############################## Training VAE ##############################
    c_nd_A, label, loss_list = vae_module.train(
        iteration=it, # Current iteration
        gmm_mu=torch.from_numpy(mu_d_A), gmm_var=torch.from_numpy(var_d_A), # mu and var estimated by Multimodal-GMM
        epoch=args.vae_iter, 
        train_loader=train_loader1, batch_size=args.batch_size, all_loader=all_loader1,
        model_dir=dir_name, agent="A"
    )
    # VAE module on Agent B
    c_nd_B, label, loss_list = vae_module.train(
        iteration=it, # Current iteration
        gmm_mu=torch.from_numpy(mu_d_B), gmm_var=torch.from_numpy(var_d_B), # mu and var estimated by Multimodal-GMM
        epoch=args.vae_iter, 
        train_loader=train_loader2, batch_size=args.batch_size, all_loader=all_loader2,
        model_dir=dir_name, agent="B"
    )
    #vae_module.plot_latent(iteration=it, all_loader=all_loader1, model_dir=dir_name, agent="A") # plot latent space of VAE on Agent A
    #vae_module.plot_latent(iteration=it, all_loader=all_loader2, model_dir=dir_name, agent="B") # plot latent space of VAE on Agent B

    K = args.category # サイン総数
    z_truth_n = label # 真のカテゴリ
    dim = len(c_nd_A[0]) # VAEの潜在変数の次元数（分散表現のカテゴリ変数の次元数）

    ############################## Initializing parameters ##############################
    # Set hyperparameters
    beta = 1.0; m_d_A = np.repeat(0.0, dim); m_d_B = np.repeat(0.0, dim) # Hyperparameters for \mu^A, \mu^B
    w_dd_A = np.identity(dim) * 0.01; w_dd_B = np.identity(dim) * 0.01 # Hyperparameters for \Lambda^A, \Lambda^B
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

    # Initializing learning parameters
    beta_hat_k_A = np.zeros(K) ;beta_hat_k_B = np.zeros(K)
    m_hat_kd_A = np.zeros((K, dim)); m_hat_kd_B = np.zeros((K, dim))
    w_hat_kdd_A = np.zeros((K, dim, dim)); w_hat_kdd_B = np.zeros((K, dim, dim))
    nu_hat_k_A = np.zeros(K); nu_hat_k_B = np.zeros(K)
    tmp_eta_nB = np.zeros((K, D)); eta_dkB = np.zeros((D, K))
    tmp_eta_nA = np.zeros((K, D)); eta_dkA = np.zeros((D, K))
    cat_liks_A = np.zeros(D); cat_liks_B = np.zeros(D)
    mu_d_A = np.zeros((D,dim)); var_d_A = np.zeros((D,dim)) 
    mu_d_B = np.zeros((D,dim)); var_d_B = np.zeros((D,dim))
    # Variables for storing the transition of each parameter
    #trace_w_in_A = [np.repeat(np.nan, D)]; trace_w_in_B = [np.repeat(np.nan, D)]
    #trace_mu_ikd_A = [mu_kd_A.copy()]; trace_mu_ikd_B = [mu_kd_B.copy()]
    #trace_lambda_ikdd_A = [lambda_kdd_A.copy()]; trace_lambda_ikdd_B = [lambda_kdd_B.copy()]
    #trace_beta_ik_A = [np.repeat(beta, K)]; trace_beta_ik_B = [np.repeat(beta, K)]
    #trace_m_ikd_A = [np.repeat(m_d_A.reshape((1, dim)), K, axis=0)]; trace_m_ikd_B = [np.repeat(m_d_B.reshape((1, dim)), K, axis=0)]
    #trace_w_ikdd_A = [np.repeat(w_dd_A.reshape((1, dim, dim)), K, axis=0)]; trace_w_ikdd_B = [np.repeat(w_dd_B.reshape((1, dim, dim)), K, axis=0)]
    #trace_nu_ik_A = [np.repeat(nu, K)]; trace_nu_ik_B = [np.repeat(nu, K)]

    iteration = args.mh_iter # M−H法のイテレーション数
    ARI_A = np.zeros((iteration)); ARI_B = np.zeros((iteration)); concidence = np.zeros((iteration))
    accept_count_AtoB = np.zeros((iteration)); accept_count_BtoA = np.zeros((iteration)) # Number of acceptation
    ############################## M-H algorithm ##############################
    print(f"M-H algorithm Start({it}): Epoch:{iteration}")
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
            pred_label_A.append(np.argmax(w_dk_A[d]))
            
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

            for d in range(D):
                mu_d_B[d] = mu_kd_B[np.argmax(w_dk[d])]
                var_d_B[d] = np.diag(np.linalg.inv(lambda_kdd_B[np.argmax(w_dk[d])]))

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

            for d in range(D):
                mu_d_A[d] = mu_kd_A[np.argmax(w_dk[d])]
                var_d_A[d] = np.diag(np.linalg.inv(lambda_kdd_A[np.argmax(w_dk[d])]))

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

        # Kappa係数の計算
        concidence[i] = np.round((a_observed - a_chance) / (1 - a_chance), 3)
        # ARIの計算
        ARI_A[i] = np.round(calc_ari(pred_label_A, z_truth_n)[0],3); ARI_B[i] = np.round(calc_ari(pred_label_B, z_truth_n)[0],3)
        # 受容回数
        accept_count_AtoB[i] = count_AtoB; accept_count_BtoA[i] = count_BtoA
        
        if i == 0 or (i+1) % 10 == 0 or i == (iteration-1): 
            print(f"=> Ep: {i+1}, A: {ARI_A[i]}, B: {ARI_B[i]}, C:{concidence[i]}, A2B:{int(accept_count_AtoB[i])}, B2A:{int(accept_count_BtoA[i])}")


        # 値を記録
        #_, w_n_A = np.where(w_dk_A == 1); _, w_n_B = np.where(w_dk_B == 1)
        #trace_w_in_A.append(w_n_A.copy()); trace_w_in_B.append(w_n_B.copy())
        #trace_mu_ikd_A.append(mu_kd_A.copy()); trace_mu_ikd_B.append(mu_kd_B.copy())
        #trace_lambda_ikdd_A.append(lambda_kdd_A.copy()); trace_lambda_ikdd_B.append(lambda_kdd_B.copy())
        #trace_beta_ik_A.append(beta_hat_k_A.copy()); trace_beta_ik_B.append(beta_hat_k_B.copy())
        #trace_m_ikd_A.append(m_hat_kd_A.copy()); trace_m_ikd_B.append(m_hat_kd_B.copy())
        #trace_w_ikdd_A.append(w_hat_kdd_A.copy()); trace_w_ikdd_B.append(w_hat_kdd_B.copy())
        #trace_nu_ik_A.append(nu_hat_k_A.copy()); trace_nu_ik_B.append(nu_hat_k_B.copy())
    
    np.save(npy_dir+'/muA_'+str(it)+'.npy', mu_kd_A); np.save(npy_dir+'/muB_'+str(it)+'.npy', mu_kd_B)
    np.save(npy_dir+'/lambdaA_'+str(it)+'.npy', lambda_kdd_A); np.save(npy_dir+'/lambdaB_'+str(it)+'.npy', lambda_kdd_B)    
    np.savetxt(log_dir+"/ariA"+str(it)+".txt", ARI_B, fmt ='%.3f'); np.savetxt(log_dir+"/ariB"+str(it)+".txt", ARI_B, fmt ='%.2f'); np.savetxt(log_dir+"/cappa"+str(it)+".txt", concidence, fmt ='%.2f')

    # 受容回数
    plt.figure()
    #plt.ylim(0,)
    plt.plot(range(0,iteration), accept_count_AtoB, marker="None", label="Accept_num:AtoB")
    plt.plot(range(0,iteration), accept_count_BtoA, marker="None", label="Accept_num:BtoA")
    plt.xlabel('iteration');plt.ylabel('Number of acceptation')
    plt.ylim(0,D)
    plt.legend()
    plt.savefig(result_dir+'/accept'+str(it)+'.png')
    #plt.show()
    plt.close()

    
    # concidence
    plt.figure()
    plt.plot(range(0,iteration), concidence, marker="None")
    plt.xlabel('iteration'); plt.ylabel('Concidence')
    plt.ylim(0,1)
    plt.title('Cappa')
    plt.savefig(result_dir+"/conf"+str(it)+".png")
    #plt.show()
    plt.close()

    # ARI
    plt.figure()
    plt.plot(range(0,iteration), ARI_A, marker="None",label="ARI_A")
    plt.plot(range(0,iteration), ARI_B, marker="None",label="ARI_B")
    plt.xlabel('iteration'); plt.ylabel('ARI')
    plt.ylim(0,1)
    plt.legend()
    plt.title('ARI')
    plt.savefig(result_dir+"/ari"+str(it)+".png")
    #plt.show()
    plt.close()
    print(f"Iteration:{it} Done:max_A: {max(ARI_A)}, max_B: {max(ARI_B)}, max_c:{max(concidence)}")