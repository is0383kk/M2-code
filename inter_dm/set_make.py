import random
from PIL import Image

num_s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for i in range(len(num_s)):
    num = num_s[i]
    num = "{:03d}".format(num)
    if num in num_s:
        print("The same file has been selected.")
        break
    dir_v = "../Multi_S/" + num + "/Vision/Hist_B.txt"
    dir_s = "../Multi_S/" + num + "/Sound/Hist_B.txt"
    dir_h = "../Multi_S/" + num + "/Haptic/Hist_B.txt"
    num_s[i] = num

    with open(dir_v, 'r') as rvf:
        sv = rvf.read()
        if i == 0:
            with open("../s_test_data/feature_vision_B.txt", "w") as wvf:
                for j in range(10):
                    wvf.write(sv)
        else:
            for j in range(10):
                with open("../s_test_data/feature_vision_B.txt", "a") as wvf:
                    wvf.write(sv)

    with open(dir_s, 'r') as rsf:
        ss = rsf.read()
        if i == 0:
            with open("../s_test_data/feature_sound_B.txt", "w") as wsf:
                for j in range(10):
                    wsf.write(ss)
        else:
            for j in range(10):
                with open("../s_test_data/feature_sound_B.txt", "a") as wsf:
                    wsf.write(ss)

    with open(dir_h, 'r') as rhf:
        sh = rhf.read()
        if i == 0:
            with open("../s_test_data/feature_haptic_B.txt", "w") as whf:
                for j in range(10):
                    whf.write(sh)
        else:
            for j in range(10):
                with open("../s_test_data/feature_haptic_B.txt", "a") as whf:
                    whf.write(sh)


print("number of dir = {}".format(num_s))
