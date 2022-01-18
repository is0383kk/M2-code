import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iteration = 200
data_num = 10

d1 = np.array([[0.0 for a in range(iteration)] for b in range(data_num)])
d2 = np.array([[0.0 for a in range(iteration)] for b in range(data_num)])
d1_mean = [0.0 for a in range(iteration)]
d2_mean = [0.0 for a in range(iteration)]
sum_1 = 0
sum_2 = 0
n1 = [0.0 for i in range(data_num)]
n2 = [0.0 for i in range(data_num)]
d1_err = [0.0 for a in range(iteration)]
d2_err = [0.0 for a in range(iteration)]

for i in range(data_num):
    #data_dir1 = '../result/Experiment3/Data' + str(i) + '/MH_VH_S_iter_200/ARI_b.csv'
    #data_dir2 = '../result/Experiment3/Data' + str(i) + '/MHn_VH_S_iter_200/ARI_b.csv'
    data_dir1 = '../result/S_Experiment2/Data' + str(i) + '/V_iter_200/ARI_a.csv'
    data_dir2 = '../result/S_Experiment2/Data' + str(i) + '/V_iter_200/ARI_b.csv'
    df1 = pd.read_csv(data_dir1, names=['num1'])
    df2 = pd.read_csv(data_dir2, names=['num1'])

    d1[i] = df1['num1']
    d2[i] = df2['num1']
    n1[i] = d1[i][iteration-1]
    n2[i] = d2[i][iteration-1]

for j in range(iteration):
    d1_std = np.array([0.0 for a in range(data_num)])
    d2_std = np.array([0.0 for a in range(data_num)])
    std_1 = 0.0
    std_2 = 0.0
    for k in range(data_num):
        sum_1 += d1[k][j]
        sum_2 += d2[k][j]
        d1_std[k] = d1[k][j]
        d2_std[k] = d2[k][j]
    d1_mean[j] = sum_1 / data_num
    d2_mean[j] = sum_2 / data_num
    std_1 = np.std(d1_std)
    std_2 = np.std(d2_std)
    d1_err[j] = round(std_1, 3)
    d2_err[j] = round(std_2, 3)
    sum_1 = 0
    sum_2 = 0

plt.errorbar(range(0,iteration), d1_mean, yerr = d1_err, fmt='none', ecolor='r', alpha=0.1, elinewidth = 1.8)
plt.errorbar(range(0,iteration), d2_mean, yerr = d2_err, fmt='none', ecolor='b', alpha=0.1, elinewidth = 1.8)
#plt.errorbar(range(0,iteration), d1_mean, yerr = d1_err, fmt='none', ecolor='#1f77b4', alpha=0.2, elinewidth = 1.8)
#plt.errorbar(range(0,iteration), d2_mean, yerr = d2_err, fmt='none', ecolor='#ff7f0e', alpha=0.2, elinewidth = 1.8)
plt.plot(range(0,iteration),d1_mean,marker="None",label="Agent A",color='r')
plt.plot(range(0,iteration),d2_mean,marker="None",label="Agent B",color='b')
#plt.plot(range(0,iteration),d1_mean,marker="None",label="communication",color='#1f77b4')
#plt.plot(range(0,iteration),d2_mean,marker="None",label="No communication",color='#ff7f0e')
plt.xlabel('iteration')
#plt.ylabel('Kappa coefficient')
plt.ylabel('ARI')
#plt.ylim(-0.01, 1.0)
plt.ylim(0.0, 1.0)
#plt.legend()
plt.legend(loc='lower right')
plt.show()
#plt.savefig("../result/Experiment/re/MH_V_iter_200.png")
#plt.savefig("../result/S_Experiment/re/B_MH_VH_S_iter_200.png")

#for k in range(data_num):
    #print("{}  {}".format(n1[k], n2[k]))
    #print("{}".format(n1[k]))

#print('\n')

#for i in range(iteration):
#    print(d1_err[i])
