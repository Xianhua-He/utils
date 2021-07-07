import os
from multiprocessing import Process
from PIL import Image
import numpy as np


# path = '/ssd/hexianhua/dataset/driver_dataset'
path = '../debug'
def read_image(data_list, s, e):
    for file in data_list[s:e]:
        try:
            img = Image.open(file)
            img = np.array(img.convert('RGB')).astype(np.float32)
            print(img.shape)
        except:
            print("failed open file: {}".format(file))
            os.remove(file)
            continue

process_num = 32
data_list = []
for root, dirs, files in os.walk(path):
        for f in files:
            name = os.path.join(root, f)
            data_list.append(name)
print(len(data_list))

try:
    process_list = []
    t_step = int(len(data_list)/process_num)
    for j in range(process_num):
        if j == process_num-1:
            s = j*t_step
            e = len(data_list)
        else:
            s = j*t_step
            e = (j+1)*t_step
        t = Process(target=read_image, args=(data_list, s, e))
        process_list.append(t)
    for i, pro in enumerate(process_list):
        pro.start()
    for pro in process_list:
        pro.join()
except TypeError:
    print('end')

