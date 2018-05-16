from frontends import model
from data_utils import triplet_generator2
from keras.callbacks import ModelCheckpoint

BASE_MODEL="INCEPTIONV3"##choose from ["INCEPTIONV3","VGG16","MOBILENET","RESNET50","XCEPTION","DENSENET"]
bs=32

facenet=model(BASE_MODEL)
print(30*"-"+"got facenet"+30*"-")
facenet.summary()

layer_name="model_2"
features_extractor=facenet.get_layer(layer_name)

train_g=triplet_generator2(features_extractor,dir_path="lfw_alig2_train",batch_size=32).flow()

for i in range(500):
  [a,p,n],_=train_g.__next__()
  facenet.fit([a,p,n],epochs=10,verbose=1,
		      callbacks=[ModelCheckpoint("models/"+BASE_MODEL+".h5",monitor="loss",mode="min")])
