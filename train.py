from frontends import model
from data_utils import triplet_generator
from keras.callbacks import ModelCheckpoint
import numpy as np

def get_distance(emb_a,emb_b):
	return np.sum(np.square(emb_a-emb_b))

BASE_MODEL="INCEPTIONV3"##choose from ["INCEPTIONV3","VGG16","MOBILENET","RESNET50","XCEPTION","DENSENET"]
bs=32

facenet=model(BASE_MODEL)
print(30*"-"+"got facenet"+30*"-")
facenet.summary()

train_g=triplet_generator(dir_path="lfw_alig2",batch_size=bs*60)
min_loss=99999999999
for i in range(500):
	this_anchor,this_positive,this_negative=train_g.next()
	facenet.fit([this_anchor,this_positive,this_negative],epochs=1,batch_size=bs)
	this_loss=facenet.evaluate([this_anchor,this_positive,this_negative])

	if this_loss<min_loss:
		print("Loss improved from "+str(min_loss)+" to "+str(this_loss))
		min_loss=this_loss
		facenet.save_weights(BASE_MODEL+".h5")

	layer_name="model_2"
	features_extractor=facenet.get_layer(layer_name)
	this_anchor_emb=features_extractor.predict(this_anchor)
	this_positive_emb=features_extractor.predict(this_positive)
	this_negative_emb=features_extractor.predict(this_negative)
	max_j=-1
	max_dist=-99999999999
	for j in bs:
		dist_p=get_distance(this_anchor_emb[j],this_positive_emb[j])
		dist_n=get_distance(this_anchor_emb[j],this_negative_emb[j])
		dist=dist_p-dist_n
		if dist>max_dist:
			max_dist=dist
			max_j=j
	facenet.fit([np.expand_dims(this_anchor[max_j],axis=0),np.expand_dims(this_positive[max_j],axis=0),np.expand_dims(this_negative[max_j],axis=0)],epochs=1,batch_size=1)
	this_loss=facenet.evaluate([this_anchor,this_positive,this_negative])

	if this_loss<min_loss:
		print("Loss improved from "+str(min_loss)+" to "+str(this_loss))
		min_loss=this_loss
		facenet.save_weights(BASE_MODEL+".h5")
