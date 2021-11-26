import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx = confusion_matrix(y_true, y_pred, labels=labels)
    #cmd = ConfusionMatrixDisplay(cmx,display_labels=None)
    #cmd.plot()
    df_cmx = pd.DataFrame(cmx, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=False)
    plt.show()
    #plt.savefig(result_dir+"/ari"+str(it)+".png")

print_cmx([0,1,2,3,4,0,1,2,3,4],[0,1,2,3,2,0,1,2,3,4])