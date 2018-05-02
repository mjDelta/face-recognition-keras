import os
import numpy as np
import shutil

def makedir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
makedir("lfw_alig2_train/")
makedir("lfw_alig2_test/")

dir_path="lfw_alig2/"
fs=os.listdir(dir_path)
np.random.shuffle(fs)


train_rate=0.95

train_fs=fs[:int(train_rate*len(fs))]
test_fs=fs[int(train_rate*len(fs)):]

for f in fs:
	if f in train_fs:
		src=os.path.join(dir_path,f)
		dst=os.path.join("lfw_alig2_train",f)
		shutil.copytree(src,dst)
	else:
		src=os.path.join(dir_path,f)
		dst=os.path.join("lfw_alig2_test",f)
		shutil.copytree(src,dst)

