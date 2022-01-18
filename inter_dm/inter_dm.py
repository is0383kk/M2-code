#!/usr/bin/env python
# -*- coding:utf-8 -*-
##########################################
#Metropolis-Hastings (Multimodal) algorithm
#Author Kazuma Furukawa

#-==============================================================================

########################################## import
import numpy as np
from sklearn import metrics
import sys
import time
sys.path.append("../lib/")
import BoF
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

start_time=time.time()

#データセットのあるディレクトリを指定

##実物体データセットのファイルパス
#dir_a_vision = "../test_data/feature_vision.txt" #データA（40カテゴリ、400データ、Vision特徴 エージェントA）
#dir_a_sound = "../test_data/feature_sound.txt"   #データA (40カテゴリ、400データ、Sound特徴 エージェントA)
#dir_a_haptic = "../test_data/feature_haptic.txt" #データA (40カテゴリ、400データ、Haptic特徴 エージェントA)

#dir_b_vision = "../test_data/feature_vision.txt" #データB（40カテゴリ、400データ、Vision特徴 エージェントB）
#dir_b_sound = "../test_data/feature_sound.txt"   #データB (40カテゴリ、400データ、Sound特徴 エージェントB)
#dir_b_haptic = "../test_data/feature_haptic.txt" #データB (40カテゴリ、400データ、Haptic特徴 エージェントB)

##人工データセットのファイルパス
#dir_a_vision = "./s_test_data/feature_vision_A.txt" #擬似データA（15カテゴリ、150データ、Vision特徴 エージェントA）
dir_a_vision = "./histogram_vA.txt" #擬似データA（15カテゴリ、150データ、Vision特徴 エージェントA）
#dir_a_sound = "./s_test_data/feature_sound_A.txt"   #擬似データA（15カテゴリ、150データ、Sound特徴 エージェントA）
#dir_a_haptic = "./s_test_data/feature_haptic_A.txt" #擬似データA（15カテゴリ、150データ、Haptic特徴 エージェントA）

dir_b_vision = "./histogram_vB.txt" #擬似データB（15カテゴリ、150データ、Vision特徴 エージェントB）
#dir_b_sound = "./s_test_data/feature_sound_B.txt"   #擬似データB（15カテゴリ、150データ、Sound特徴 エージェントB）
#dir_b_haptic = "./s_test_data/feature_haptic_B.txt" #擬似データB（15カテゴリ、150データ、Haptic特徴 エージェントB）


#print(len())
#data_set_num = 1272  #データセットの数
data_set_num = 1 #データセットの数
"""
DATA_NUM = 150 #データ(物体)の数
WORD_DIM = 15   #記号(サイン)数
CONCEPT_DIM = 15  #カテゴリ数
"""
DATA_NUM = 1272
WORD_DIM = 10   #記号(サイン)数
CONCEPT_DIM = 10  #カテゴリ数

iteration = 100  #イテレーション(反復回数)

#ハイパーパラメータ設定
beta_c_a_vision = 0.001
beta_c_a_sound = 0.001
beta_c_a_haptic = 0.001
alpha_w_a = 0.01

beta_c_b_vision = 0.001
beta_c_b_sound = 0.001
beta_c_b_haptic = 0.001
alpha_w_b = 0.01

#重みの設定
w_a_v = 1.0
w_a_s = 1.0
w_a_h = 1.0

w_b_v = 1.0
w_b_s = 1.0
w_b_h = 1.0

w_a_w = 1.0
w_b_w = 1.0

#エージェントごとのカテゴリ数
concept_num_a = CONCEPT_DIM
concept_num_b = CONCEPT_DIM

#===============================================================================

def softmax(arr, axis=0):
    arr = np.rollaxis(arr, axis)
    vmax = arr.max(axis=0)
    out = np.exp(arr-vmax) / np.sum(np.exp(arr-vmax), axis=0)

    return out

def new_feature_data_read(directory):
    all_feature = []
    f = directory
    feat = np.loadtxt(f)
    all_feature.append(feat)

    return all_feature

def Multi_prob(data, phi):
    phi_log = np.log(phi)
    prob = data.dot(phi_log.T)

    return prob

#===============================================================================

#Metropolis-Hasting algorithm
def Metropolis_Hasting():

    #データ読み込み
    ##初期化
    print("Parameter Initialization 1")
    feature_set_a_vision = []
    #feature_set_a_sound = []
    #feature_set_a_haptic = []
    feature_set_b_vision = []
    #feature_set_b_sound = []
    #feature_set_b_haptic = []
    word_set_a = []
    word_set_b = []

    #テストデータ
    #word_set_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39]
    
    word_set_t = np.loadtxt("label.txt")
    #---------------------------------------------------------------------------

    #各物体のカテゴリ決定(初期化)
    c_w_a = [1000 for n in range(DATA_NUM)]
    c_w_b = [1000 for n in range(DATA_NUM)]

    #物体ごとの記号(サイン)当て
    word_a = np.zeros((DATA_NUM, WORD_DIM))
    word_b = np.zeros((DATA_NUM, WORD_DIM))

    initial_word_a = np.zeros((DATA_NUM, WORD_DIM))
    initial_word_b = np.zeros((DATA_NUM, WORD_DIM))

    #ARI, Kappaの結果
    ARI_a = np.zeros((iteration))
    ARI_b = np.zeros((iteration))
    ARI_ab = np.zeros((iteration))
    concidence = np.zeros((iteration))

    #---------------------------------------------------------------------------

    #特徴データ読み込み
    print("Reading Data")
    feature_set_a_vision = new_feature_data_read(dir_a_vision)
    #print("feature_set_a_vision : ", feature_set_a_vision)
    #feature_set_a_sound = new_feature_data_read(dir_a_sound)
    #feature_set_a_haptic = new_feature_data_read(dir_a_haptic)

    feature_set_b_vision = new_feature_data_read(dir_b_vision)
    #feature_set_b_sound = new_feature_data_read(dir_b_sound)
    #feature_set_b_haptic = new_feature_data_read(dir_b_haptic)

    #各特徴の次元数
    FEATURE_DIM_a_vision = len(feature_set_a_vision[0][0])
    #print("FEATURE_DIM_a_vision : ", FEATURE_DIM_a_vision)
    #FEATURE_DIM_a_sound = len(feature_set_a_sound[0][0])
    #FEATURE_DIM_a_haptic = len(feature_set_a_haptic[0][0])

    FEATURE_DIM_b_vision = len(feature_set_b_vision[0][0])
    #FEATURE_DIM_b_sound = len(feature_set_b_sound[0][0])
    #FEATURE_DIM_b_haptic = len(feature_set_b_haptic[0][0])

    #===========================================================================

    #多項分布パラメータphi,theta設定
    ##初期化
    print("Parameter Initialization 2")
    #各カテゴリにどの特徴が振られるか
    phi_f_e_a_vision = np.array([[[float(1.0)/FEATURE_DIM_a_vision for i in range(FEATURE_DIM_a_vision)] for j in range(concept_num_a)] for k in range(data_set_num)])
    #print("phi_f_e_a_vision", phi_f_e_a_vision.shape)
    #phi_f_e_a_sound = np.array([[[float(1.0)/FEATURE_DIM_a_sound for i in range(FEATURE_DIM_a_sound)] for j in range(concept_num_a)] for k in range(data_set_num)])
    #phi_f_e_a_haptic = np.array([[[float(1.0)/FEATURE_DIM_a_haptic for i in range(FEATURE_DIM_a_haptic)] for j in range(concept_num_a)] for k in range(data_set_num)])

    phi_f_e_b_vision = np.array([[[float(1.0)/FEATURE_DIM_b_vision for i in range(FEATURE_DIM_b_vision)] for j in range(concept_num_b)] for k in range(data_set_num)])
    #phi_f_e_b_sound = np.array([[[float(1.0)/FEATURE_DIM_b_sound for i in range(FEATURE_DIM_b_sound)] for j in range(concept_num_b)] for k in range(data_set_num)])
    #phi_f_e_b_haptic = np.array([[[float(1.0)/FEATURE_DIM_b_haptic for i in range(FEATURE_DIM_b_haptic)] for j in range(concept_num_b)] for k in range(data_set_num)])

    #各記号(サイン)にどのカテゴリが振られるか
    theta_w_e_a = np.array([[[float(1.0)/concept_num_a for i in range(concept_num_a)] for j in range(WORD_DIM)] for k in range(data_set_num)])
    theta_w_e_b = np.array([[[float(1.0)/concept_num_b for i in range(concept_num_b)] for j in range(WORD_DIM)] for k in range(data_set_num)])

    #カテゴリ番号
    class_choice_a = [dc for dc in range(concept_num_a)]
    class_choice_b = [dc for dc in range(concept_num_b)]

    #サイン番号
    word_choice = [dc for dc in range(WORD_DIM)] # [0,1,2,...,14]
    #print(word_choice)

    #記号(サイン)に対する乱数ベクトルの初期化
    rand_set_a = []
    for d in range(DATA_NUM):
        rand_a = np.random.randint(0, WORD_DIM)
        rand_set_a.append(rand_a)
    #print("rand_set_a",rand_set_a)

    rand_set_b = []
    for d in range(DATA_NUM):
        rand_b = np.random.randint(0, WORD_DIM)
        rand_set_b.append(rand_b)

    for d in range(DATA_NUM):
        word_a[d][rand_set_a[d]] = 1
        initial_word_a[d][rand_set_a[d]] = 1

        word_b[d][rand_set_b[d]] = 1
        initial_word_b[d][rand_set_b[d]] = 1
    #print("word_a",word_a[0])

    #===========================================================================
    #print(theta_w_e_a)
    for e in range(data_set_num): #1回
        #エージェントAの各データへのカテゴリの割り当て
        class_count_e_set_a = []
        theta_w_log_a = np.log(theta_w_e_a[0])
        for d in range(DATA_NUM):
            class_count_a = [0.0 for i in range(concept_num_a)]
            multi_prob_set_a = np.zeros((concept_num_a),dtype=float)

            ##視覚情報
            if w_a_v > 0.0:
                for i in range(concept_num_a):
                    multi_prob_set_a[i] += w_a_v * Multi_prob(feature_set_a_vision[0][d], phi_f_e_a_vision[0][i])
            #print("multi_prob_set_b : ", multi_prob_set_a)
            #print("feature_set_a_vision : ", feature_set_a_vision)

            ##記号(サイン)情報
            for j in range(WORD_DIM):
                if word_a[d][j] == 1:
                    for k in range(concept_num_a):
                        multi_prob_set_a[k] += w_a_w * theta_w_log_a[j][k]

            multi_prob_set_a = softmax(multi_prob_set_a)
            #print("multi_prob_set_a : ", multi_prob_set_a)
            c_w_a[d] = np.random.choice(class_choice_a, p=multi_prob_set_a)
            #print(c_w_a[d])
            class_count_a[c_w_a[d]] += 1.0
            class_count_e_set_a.append(class_count_a)

        #データに割り当てられたカテゴリから, phi_f_e_a_* を計算
        for c in range(concept_num_a):
            feat_e_c_a_vision = []

            for d in range(DATA_NUM):
                if c_w_a[d] == c:
                    feat_e_c_a_vision.append(feature_set_a_vision[0][d])
                    
            total_feat_e_a_vision = BoF.bag_of_feature(feat_e_c_a_vision, FEATURE_DIM_a_vision)
            total_feat_e_a_vision = total_feat_e_a_vision + beta_c_a_vision

            phi_f_e_a_vision[0][c] = np.random.dirichlet(total_feat_e_a_vision) + 1e-100

        #データに割り当てられたカテゴリから, theta_w_a を計算
        for w in range(WORD_DIM):
            c_e_w_a = []
            for d in range(DATA_NUM):
                if word_a[d][w] == 1:
                    c_e_w_a.append(class_count_e_set_a[d])
            total_c_e_a = BoF.bag_of_feature(c_e_w_a, concept_num_a)
            total_c_e_a = total_c_e_a + alpha_w_a

            theta_w_e_a[0][w] = np.random.dirichlet(total_c_e_a) + 1e-100

        #エージェントBの各データへのカテゴリの割り当て
        class_count_e_set_b = []
        theta_w_log_b = np.log(theta_w_e_b[0])
        for d in range(DATA_NUM):
            class_count_b = [0.0 for i in range(concept_num_b)]
            multi_prob_set_b = np.zeros((concept_num_b),dtype=float)

            ##視覚情報
            if w_b_v > 0.0:
                for i in range(concept_num_b):
                    multi_prob_set_b[i] += w_b_v * Multi_prob(feature_set_b_vision[0][d], phi_f_e_b_vision[0][i])

            ##記号(サイン)情報
            for j in range(WORD_DIM):
                if word_b[d][j] == 1:
                    for k in range(concept_num_b):
                        multi_prob_set_b[k] += w_b_w * theta_w_log_b[j][k]
            #print("multi_prob_set_b", multi_prob_set_b.shape)
            multi_prob_set_b = softmax(multi_prob_set_b)
            c_w_b[d] = np.random.choice(class_choice_b, p=multi_prob_set_b)
            class_count_b[c_w_b[d]] += 1.0
            class_count_e_set_b.append(class_count_b)

        #データに割り当てられたカテゴリから, phi_f_e_b_* を計算
        for c in range(concept_num_b):
            feat_e_c_b_vision = []

            for d in range(DATA_NUM):
                if c_w_b[d] == c:
                    feat_e_c_b_vision.append(feature_set_b_vision[0][d])

            total_feat_e_b_vision = BoF.bag_of_feature(feat_e_c_b_vision, FEATURE_DIM_b_vision)
            total_feat_e_b_vision = total_feat_e_b_vision + beta_c_b_vision

            phi_f_e_b_vision[0][c] = np.random.dirichlet(total_feat_e_b_vision) + 1e-100

        #データ割り当てられたカテゴリから, theta_w_b を計算
        for w in range(WORD_DIM):
            c_e_w_b = []
            for d in range(DATA_NUM):
                if word_b[d][w] == 1:
                    c_e_w_b.append(class_count_e_set_b[d])
            total_c_e_b = BoF.bag_of_feature(c_e_w_b, concept_num_b)
            total_c_e_b = total_c_e_b + alpha_w_b

            theta_w_e_b[0][w] = np.random.dirichlet(total_c_e_b) + 1e-100

    #===========================================================================
    """
    以下M-H法
    """
    #パラメータ推定
    for iter in range(100):
        print("-------------iteration" + repr(iter)+ "-------------")
        for e in range(data_set_num):
            ##エージェントAから記号(サイン)のサンプリング
            new_word_a_set = []
            for d in range(DATA_NUM):
                word_multi_prob_set_a = np.zeros(WORD_DIM, dtype=float)
                total_phi_a = 0.0
                for w in range(WORD_DIM):
                    total_phi_a += theta_w_e_a[0][w][c_w_a[d]]
                    word_multi_prob_set_a[w] = theta_w_e_a[0][w][c_w_a[d]]
                    #print("total_phi_a->", total_phi_a)
                    #print("word_multi_prob_set_a[w]", word_multi_prob_set_a[w])
                #print("total_phi_a : ", total_phi_a)
                word_multi_prob_set_a = word_multi_prob_set_a / total_phi_a
                #print(f"word_multi_prob_set_a={word_multi_prob_set_a} / total_phi_a={total_phi_a}")
                #print("word_multi_prob_set_a : ", word_multi_prob_set_a)
                #print("sum(word_multi_prob_set_a) : ", sum(word_multi_prob_set_a))

                new_word_a_set.append(np.random.choice(word_choice, p=word_multi_prob_set_a))
                #print("new_word_a_set", new_word_a_set) # [3, 4,...]のようにカテゴリが割り振られる

            ##A提案の記号(サイン)の取捨選択
            for d in range(DATA_NUM):
                word_multi_prob_b = np.zeros(1, dtype=float)
                word_multi_prob_b = theta_w_e_b[0][rand_set_b[d]][c_w_b[d]]
                #print("word_multi_prob_b : ", word_multi_prob_b)
                new_word_multi_prob_b = np.zeros(1, dtype=float)
                new_word_multi_prob_b = theta_w_e_b[0][new_word_a_set[d]][c_w_b[d]]
                #print("new_word_multi_prob_b : ", new_word_multi_prob_b)
                #print("word_multi_prob_b : ", word_multi_prob_b)
                judge_r = new_word_multi_prob_b / word_multi_prob_b   
                #print("judge_r : ",judge_r)
                judge_r = min(1, judge_r)
                rand_u = np.random.rand()
                
                if (judge_r >= rand_u):
                    rand_set_b[d] = new_word_a_set[d]
                    for i in range(WORD_DIM):
                        word_b[d][i] = 0
                    word_b[d][new_word_a_set[d]] = 1
                
            #print("word_b : ", word_b)
            ##エージェントBの各データへのカテゴリの再割り当て
            class_count_e_set_b = []
            theta_w_log_b = np.log(theta_w_e_b[0])
            #print("theta_w_log_b : ", theta_w_log_b)
            for d in range(DATA_NUM):
                class_count_b = [0.0 for i in range(concept_num_b)]
                multi_prob_set_b = np.zeros((concept_num_b),dtype=float)

                ##視覚情報
                if w_b_v > 0.0:
                    for i in range(concept_num_b):
                        multi_prob_set_b[i] += w_b_v * Multi_prob(feature_set_b_vision[0][d], phi_f_e_b_vision[0][i])
                
                ##記号(サイン)情報
                for j in range(WORD_DIM):
                    if word_b[d][j] == 1:
                        for k in range(concept_num_b):
                            multi_prob_set_b[k] += w_b_w * theta_w_log_b[j][k]
                #print("multi_prob_set_b : ", multi_prob_set_b.shape)
                multi_prob_set_b = softmax(multi_prob_set_b)
                c_w_b[d] = np.random.choice(class_choice_b, p=multi_prob_set_b)
                class_count_b[c_w_b[d]] += 1.0
                class_count_e_set_b.append(class_count_b)

            ##データに割り当てられたカテゴリから, phi_f_b_* の再計算
            for c in range(concept_num_b):
                feat_e_c_b_vision = []

                for d in range(DATA_NUM):
                    if c_w_b[d] == c:
                        feat_e_c_b_vision.append(feature_set_b_vision[0][d])

                total_feat_e_b_vision = BoF.bag_of_feature(feat_e_c_b_vision, FEATURE_DIM_b_vision)
                total_feat_e_b_vision = total_feat_e_b_vision + beta_c_b_vision

                phi_f_e_b_vision[0][c] = np.random.dirichlet(total_feat_e_b_vision) + 1e-100

            ##データに割り当てられたカテゴリから, theta_w_b の再計算
            for w in range(WORD_DIM):
                c_e_w_b = []
                for d in range(DATA_NUM):
                    if word_b[d][w] == 1:
                        c_e_w_b.append(class_count_e_set_b[d])
                total_c_e_b = BoF.bag_of_feature(c_e_w_b, concept_num_b)
                total_c_e_b = total_c_e_b + alpha_w_b

                theta_w_e_b[0][w] = np.random.dirichlet(total_c_e_b) + 1e-100

            ##エージェントBから記号(サイン)のサンプリング
            new_word_b_set = []
            for d in range(DATA_NUM):
                word_multi_prob_set_b = np.zeros(WORD_DIM, dtype=float)
                total_phi_b = 0.0
                for w in range(WORD_DIM):
                    total_phi_b += theta_w_e_b[0][w][c_w_b[d]]
                    word_multi_prob_set_b[w] = theta_w_e_b[0][w][c_w_b[d]]

                word_multi_prob_set_b = word_multi_prob_set_b / total_phi_b

                new_word_b_set.append(np.random.choice(word_choice, p=word_multi_prob_set_b))

            ##B提案の記号(サイン)の取捨選択
            for d in range(DATA_NUM):
                word_multi_prob_a = np.zeros(1, dtype=float)
                word_multi_prob_a = theta_w_e_a[0][rand_set_a[d]][c_w_a[d]]

                new_word_multi_prob_a = np.zeros(1, dtype=float)
                new_word_multi_prob_a = theta_w_e_a[0][new_word_b_set[d]][c_w_a[d]]

                judge_r = new_word_multi_prob_a / word_multi_prob_a
                judge_r = min(1, judge_r)
                rand_u = np.random.rand()
                
                if (judge_r >= rand_u):
                    rand_set_a[d] = new_word_b_set[d]
                    for i in range(WORD_DIM):
                        word_a[d][i] = 0
                    word_a[d][new_word_b_set[d]] = 1
                
            ##エージェントAの各データへのカテゴリの再割り当て
            class_count_e_set_a = []
            theta_w_log_a = np.log(theta_w_e_a[0])
            for d in range(DATA_NUM):
                class_count_a = [0.0 for i in range(concept_num_a)]
                multi_prob_set_a = np.zeros((concept_num_a),dtype=float)

                ##視覚情報
                if w_a_v > 0.0:
                    for i in range(concept_num_a):
                        multi_prob_set_a[i] += w_a_v * Multi_prob(feature_set_a_vision[0][d], phi_f_e_a_vision[0][i])

                ##記号(サイン)情報
                for j in range(WORD_DIM):
                    if word_a[d][j] == 1:
                        for k in range(concept_num_a):
                            multi_prob_set_a[k] += w_a_w * theta_w_log_a[j][k]

                multi_prob_set_a = softmax(multi_prob_set_a)
                c_w_a[d] = np.random.choice(class_choice_a, p=multi_prob_set_a)
                class_count_a[c_w_a[d]] += 1.0
                class_count_e_set_a.append(class_count_a)

            ##データに割り当てられたカテゴリから, phi_f_a_* の再計算
            for c in range(concept_num_a):
                feat_e_c_a_vision = []

                for d in range(DATA_NUM):
                    if c_w_a[d] == c:
                        feat_e_c_a_vision.append(feature_set_a_vision[0][d])

                total_feat_e_a_vision = BoF.bag_of_feature(feat_e_c_a_vision, FEATURE_DIM_a_vision)
                total_feat_e_a_vision = total_feat_e_a_vision + beta_c_a_vision

                phi_f_e_a_vision[0][c] = np.random.dirichlet(total_feat_e_a_vision) + 1e-100

            ##データに割り当てられたカテゴリから, theta_w_a の再計算
            for w in range(WORD_DIM):
                c_e_w_a = []
                for d in range(DATA_NUM):
                    if word_a[d][w] == 1:
                        c_e_w_a.append(class_count_e_set_a[d])
                total_c_e_a = BoF.bag_of_feature(c_e_w_a, concept_num_a)
                total_c_e_a = total_c_e_a + alpha_w_a

                theta_w_e_a[0][w] = np.random.dirichlet(total_c_e_a) + 1e-100

            ##評価値計算
            sum_same_w = 0.0
            a_chance = 0.0
            prob_w = [0.0 for i in range(WORD_DIM)]
            w_count_a = [0.0 for i in range(WORD_DIM)]
            w_count_b = [0.0 for i in range(WORD_DIM)]

            for d in range(DATA_NUM):
                if rand_set_a[d] == rand_set_b[d]:
                    sum_same_w += 1

                for w in range(WORD_DIM):
                    if rand_set_a[d] == w:
                        w_count_a[w] += 1
                    if rand_set_b[d] == w:
                        w_count_b[w] += 1

            for w in range(WORD_DIM):
                prob_w[w] = (w_count_a[w] / DATA_NUM) * (w_count_b[w] / DATA_NUM)
                a_chance += prob_w[w]
            a_observed = (sum_same_w / DATA_NUM)

            ###Kappa係数の計算
            concidence[iter] = np.round((a_observed - a_chance) / (1 - a_chance), 3)

            ###ARIの計算
            ARI_a[iter] = np.round(metrics.adjusted_rand_score(word_set_t, c_w_a), 3)
            ARI_b[iter] = np.round(metrics.adjusted_rand_score(word_set_t, c_w_b), 3)
            ARI_ab[iter] = np.round(metrics.adjusted_rand_score(c_w_a, c_w_b), 3)
            #print("c_w_a : ", c_w_a)
            #print("c_w_b : ", len(c_w_b))

            ###confusion_matrixの計算
            #if iter == 0:
            #    initial_confusion_matrix_a = metrics.confusion_matrix(word_set_t, c_w_a)
            #    initial_confusion_matrix_b = metrics.confusion_matrix(word_set_t, c_w_b)
            #if iter == (iteration - 1):
            #    final_confusion_matrix_a = metrics.confusion_matrix(word_set_t, c_w_a)
            #    final_confusion_matrix_b = metrics.confusion_matrix(word_set_t, c_w_b)

            ###評価値の表示
            print('ARI_a = ', ARI_a[iter])
            print('ARI_b = ', ARI_b[iter])
            print('ARI_ab = ', ARI_ab[iter])
            print('concidence = ', concidence[iter])
    #print(word_set_t)

    #===========================================================================

    #データ保存
    today = datetime.date.today()
    todaydetail = datetime.datetime.today()

    ##出力ファイルパスの設定
    Out_put_dir="./result" #確認or練習用
    #Out_put_dir = "../result/Experiment/Data0/MH_S_iter_{}".format(iteration)
    #Out_put_dir = "../result/S_Experiment/Data0/MH_V_iter_{}".format(iteration)

    ##処理時間計算
    finish_time = time.time() - start_time
    f = open(Out_put_dir + "/time.txt", "w")
    f.write("time: " + repr(finish_time) + "seconds.")
    f.close()

    ##環境変数保存
    f = open(Out_put_dir + "/Parameter.txt", "w")
    f.write("Iteration: " + repr(iteration) +
        "\nDATA_NUM: " + repr(DATA_NUM) +
        "\nWORD_DIM: "+ repr(WORD_DIM) +
        "\nbeta_c_a_vision" + repr(beta_c_a_vision) +
        "\nalpha_w_a" + repr(alpha_w_a) +
        "\nbeta_c_b_vision" + repr(beta_c_b_vision) +
        "\nalpha_w_b" + repr(alpha_w_b) +
        "\nconcept_num_a" + repr(concept_num_a) +
        "\nconcept_num_b" + repr(concept_num_b)
        )
    f.close()

    #np.savetxt(Out_put_dir+"/initial_word_a.csv",initial_word_a)
    #np.savetxt(Out_put_dir+"/initial_word_b.csv",initial_word_b)
    #np.savetxt(Out_put_dir+"/final_word_a.csv",word_a)
    #np.savetxt(Out_put_dir+"/final_word_b.csv",word_b)
    np.savetxt(Out_put_dir+"/ARI_a.csv",ARI_a)
    np.savetxt(Out_put_dir+"/ARI_b.csv",ARI_b)
    np.savetxt(Out_put_dir+"/ARI_ab.csv",ARI_ab)
    np.savetxt(Out_put_dir+"/concidence.csv",concidence)

    with open(Out_put_dir+'/phi_c_a_vision.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerows(phi_f_e_a_vision[0]) # 2次元配列も書き込める
    with open(Out_put_dir+'/phi_c_b_vision.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerows(phi_f_e_b_vision[0]) # 2次元配列も書き込める
    with open(Out_put_dir+'/theta_w_a.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerows(theta_w_e_a[0]) # 2次元配列も書き込める
    with open(Out_put_dir+'/theta_w_b.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerows(theta_w_e_b[0]) # 2次元配列も書き込める

    #Output the graph
    ##Kappa係数(エージェントAとエージェントBの記号(サイン)の一致度)のグラフ化
    df = pd.read_csv(Out_put_dir+'/concidence.csv', names=['num1'])
    plt.plot(range(0,iteration),df['num1'],marker="None")
    plt.xlabel('iteration')
    plt.ylabel('concidence')
    plt.savefig(Out_put_dir+"/concidence.png")
    #plt.show()

    ##エージェントA(カテゴリ)とエージェントB(カテゴリ)のARI結果のグラフ化
    #df = pd.read_csv(Out_put_dir+'/ARI_ab.csv', names=['num1'])
    #plt.plot(range(0,iteration),df['num1'],marker="None")
    #plt.xlabel('iteration')
    #plt.ylabel('ARI_ab')
    #plt.savefig(Out_put_dir+"/ARI_ab.png")
    #plt.show()

    ##エージェントA(カテゴリ)と正解ラベルのARI結果のグラフ化
    df = pd.read_csv(Out_put_dir+'/ARI_a.csv', names=['num1'])
    plt.plot(range(0,iteration),df['num1'],marker="None")
    plt.xlabel('iteration')
    plt.ylabel('ARI_a')
    plt.savefig(Out_put_dir+"/ARI_a.png")
    #plt.show()

    ##エージェントB(カテゴリ)と正解ラベルのARI結果のグラフ化
    df = pd.read_csv(Out_put_dir+'/ARI_b.csv', names=['num1'])
    plt.plot(range(0,iteration),df['num1'],marker="None")
    plt.xlabel('iteration')
    plt.ylabel('ARI_b')
    plt.savefig(Out_put_dir+"/ARI_b.png")
    #plt.show()

    ##実行前と実行後のconfusion_matrixのヒートマップ表示
    #sns.set(font_scale=1.5)
    #plt.figure(figsize = (10,7))
    #sns.heatmap(initial_confusion_matrix_a, annot=False, cmap="Greys", vmin=0, vmax=10) #annot=True
    #plt.savefig(Out_put_dir+"/initial_confusion_matrix_a.png")
    #plt.show()

    #sns.set(font_scale=1.5)
    #plt.figure(figsize = (10,7))
    #sns.heatmap(initial_confusion_matrix_b, annot=False, cmap="Greys", vmin=0, vmax=10) #annot=True
    #plt.savefig(Out_put_dir+"/initial_confusion_matrix_b.png")
    #plt.show()

    #sns.set(font_scale=1.5)
    #plt.figure(figsize = (10,7))
    #sns.heatmap(final_confusion_matrix_a, annot=False, cmap="Greys", vmin=0, vmax=10) #annot=True
    #plt.savefig(Out_put_dir+"/final_confusion_matrix_a.png")
    #plt.show()

    #sns.set(font_scale=1.5)
    #plt.figure(figsize = (10,7))
    #sns.heatmap(final_confusion_matrix_b, annot=False, cmap="Greys", vmin=0, vmax=10) #annot=True
    #plt.savefig(Out_put_dir+"/final_confusion_matrix_b.png")
    #plt.show()

if __name__ == '__main__':
    Metropolis_Hasting()
