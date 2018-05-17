import numpy as np
from frontends import model
from scipy.misc import imread
import os
from sklearn.metrics import f1_score,accuracy_score
import matplotlib.pyplot as plt 

def get_distance(emb_a,emb_b):
	return np.sum(np.square(emb_a-emb_b))

BASE_MODEL="INCEPTIONV3"##choose from ["INCEPTIONV3","VGG16","MOBILENET","RESNET50","XCEPTION","DENSENET"]
model_path=BASE_MODEL+".h5"
dir_path="lfw_alig_small"

facenet=model(BASE_MODEL)
print(facenet.summary())
facenet.load_weights(model_path)
print("-"*30+"load model completed"+30*"-")

layer_name="model_2"
features_extractor=facenet.get_layer(layer_name)
print(features_extractor.summary())

names=[]
faces=[]


for face_dir in os.listdir(dir_path):
	full_face_dir=os.path.join(dir_path,face_dir)
	for face_name in os.listdir(full_face_dir):
		names.append(face_dir)
		faces.append(imread(os.path.join(full_face_dir,face_name))/255.)
faces=np.array(faces)
faces=faces.reshape(len(faces),96,96,3)
embs=features_extractor.predict(faces)

distances=[]
trues=[]
for i in range(len(embs)):
	for j in range(i+1,len(embs)):
		d=get_distance(embs[i],embs[j])
		distances.append(d)
		if names[i]==names[j]:
			trues.append(1)
		else:
			trues.append(0)

thresholds=np.arange(0.1,10.,0.01)

f1_scores=[f1_score(trues,distances<t) for t in thresholds]
acc_scores=[accuracy_score(trues,distances<t) for t in thresholds]

max_idx=np.argmax(f1_scores)

plt.plot(thresholds,f1_scores,"r-")
plt.plot(thresholds,acc_scores,"b-")
plt.axvline(x=thresholds[max_idx],linestyle="--",lw=1)

plt.show()
