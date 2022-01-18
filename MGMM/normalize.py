import numpy as np
import matplotlib.pyplot as plt

org_data = np.loadtxt("histogram_vB.txt")*10
print(org_data[0], len(org_data), len(org_data[0]))
data_list = []
for i in range(len(org_data)):
    hist, edges = np.histogram(org_data[i], bins=50, density=True)
    #print(hist)
    w = edges[1] - edges[0]
    hist = hist * w
    data_list.append(hist)
    print( f"hist :{hist}, SUM {sum(hist)} ")
np.savetxt("dataB.txt", data_list)