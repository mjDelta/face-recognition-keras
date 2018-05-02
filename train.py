from frontends import model
from data_utils import triplet_generator
from keras.callbacks import ModelCheckpoint

BASE_MODEL="INCEPTIONV3"##choose from ["INCEPTIONV3","VGG16","MOBILENET","RESNET50","XCEPTION","DENSENET"]

facenet=model(BASE_MODEL)

print(30*"-"+"got facenet"+30*"-")
facenet.summary()

train_g=triplet_generator(dir_path="lfw_alig2",batch_size=16)
facenet.fit_generator(train_g,epochs=500,steps_per_epoch=115,
											callbacks=[ModelCheckpoint(BASE_MODEL+".h5",monitor="val_loss",mode="min")])