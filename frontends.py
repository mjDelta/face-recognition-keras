from keras.layers import Dense,Flatten,Input,Layer
from keras.models import Model
from backends import inceptionv3,vgg16,mobilenetv2
import keras.backend as K

alpha=0.2

class TripletLossLayer(Layer):
	def __init__(self, alpha, **kwargs):
		self.alpha = alpha
		super(TripletLossLayer, self).__init__(**kwargs)

	def triplet_loss(self, inputs):
		a, p, n = inputs
		p_dist = K.sum(K.square(a - p), axis=-1)
		n_dist = K.sum(K.square(a - n), axis=-1)
		return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

	def call(self, inputs):
		loss = self.triplet_loss(inputs)
		self.add_loss(loss)
		return loss


def triplet_loss(y_true,y_pred):
	a,p,n=y_pred[0],y_pred[1],y_pred[2]

	p_dist=K.sum(K.square(a-p),axis=-1)
	n_dist=K.sum(K.square(a-n),axis=-1)

	base_dist=p_dist-n_dist+alpha

	loss=K.sum(K.maximum(base_dist,0))

	return loss


def base_model(BASE_MODEL):
	input_=Input(shape=(96,96,3))
	if BASE_MODEL=="INCEPTIONV3":
		backend=inceptionv3()
		print(30*"-"+"got inceptionv3 as the backend"+30*"-")
	elif BASE_MODEL=="VGG16":
		backend=vgg16()
		print(30*"-"+"got vgg16 as the backend"+30*"-")
	elif BASE_MODEL=="MOBILENET":
		backend=mobilenetv2()
		print(30*"-"+"got mobilenetv2 as the backend"+30*"-")


	features_temp=backend(input_)
	flatten=Flatten()(features_temp)
	dense=Dense(128)(flatten)

	model=Model(input_,dense)
	print(model.summary())
	return model

def model(BASE_MODEL):
	bs_model=base_model(BASE_MODEL)

	input_a=Input(shape=(96,96,3))
	input_p=Input(shape=(96,96,3))
	input_n=Input(shape=(96,96,3))

	embedding_a=bs_model(input_a)
	embedding_p=bs_model(input_p)
	embedding_n=bs_model(input_n)

	

	triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([embedding_a, embedding_p, embedding_n])


	model=Model([input_a,input_p,input_n],triplet_loss_layer)

	model.compile(optimizer="adam",loss=None)

	return model

def triplet_loss_test():
    with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
                  tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
                  tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
        loss = triplet_loss(y_true, y_pred)

        print("loss = " + str(loss.eval()))


if __name__=="__main__":
	BASE_MODEL="MOBILENET"
	model=model(BASE_MODEL)
	model.summary()

