import glob
import os
 
# 拡張子.txtのファイルを取得する
K = 6
path = './train10_png/'+str(K)+'/*.png'
i = 1
 
# txtファイルを取得する
flist = glob.glob(path)
print('変更前')
print(flist)
 
# ファイル名を一括で変更する
for file in flist:
  os.rename(file, './train10_png/'+str(K)+'/'+str(K)+'_' + str(i) + '.png')
  i+=1
 
list = glob.glob(path)
print('変更後')
print(list)