from keras.layers import Dense,Conv2D,BatchNormalization,MaxPooling2D,AveragePooling2D,Activation,Input,concatenate,Flatten
from keras.models import Model
import keras.backend as K
from my_conv_utils import DepthwiseConv2D
from keras.layers import add
ROW_SIZE=96
COL_SIZE=96
ALPHA=0.125
def inceptionv3():
	def conv2d_bn_(x,kernels,kernel_size,strides=(1,1),padding="same"):
		x=Conv2D(int(ALPHA*kernels),kernel_size,strides=strides,padding=padding)(x)
		x=BatchNormalization()(x)
		x=Activation("relu")(x)
		return x
	def block1(x):##1*1,1*1+5*5,1*1+3*3+3*3,avgpool+1*1
		branch1_1x1=conv2d_bn_(x,64,(1,1))

		branch2_1x1=conv2d_bn_(x,48,(1,1))
		branch2_5x5=conv2d_bn_(branch2_1x1,64,((5,5)))

		branch3_1x1=conv2d_bn_(x,64,(1,1))
		branch3_3x3=conv2d_bn_(branch3_1x1,96,(3,3))
		branch3_3x3=conv2d_bn_(branch3_3x3,96,(3,3))

		branch4_pool=AveragePooling2D((3,3),strides=(1,1),padding="same")(x)
		branch4_pool=conv2d_bn_(branch4_pool,32,(1,1))

		x=concatenate([branch1_1x1,branch2_5x5,branch3_3x3,branch4_pool])
		return x
	def block2(x,temp_kernels):
		branch1_1x1=conv2d_bn_(x,192,(1,1))

		branch2_1x1=conv2d_bn_(x,temp_kernels,(1,1))
		branch2_1x7=conv2d_bn_(branch2_1x1,temp_kernels,(1,7))
		branch2_7x1=conv2d_bn_(branch2_1x7,192,(7,1))

		branch3_1x1=conv2d_bn_(x,temp_kernels,(1,1))
		branch3_7x1=conv2d_bn_(branch3_1x1,temp_kernels,(7,1))
		branch3_1x7=conv2d_bn_(branch3_7x1,temp_kernels,(1,7))
		branch3_7x1=conv2d_bn_(branch3_1x7,temp_kernels,(7,1))
		branch3_1x7=conv2d_bn_(branch3_7x1,192,(1,7))
		
		branch4_pool=AveragePooling2D((3,3),strides=(1,1),padding="same")(x)
		branch4_pool=conv2d_bn_(branch4_pool,192,(1,1))

		x=concatenate([branch1_1x1,branch2_7x1,branch3_1x7,branch4_pool])
		return x
	def block3(x):
		branch1_1x1=conv2d_bn_(x,320,(1,1))

		branch2_1x1=conv2d_bn_(x,384,(1,1))
		branch2_1x3=conv2d_bn_(branch2_1x1,384,(1,3))
		branch2_3x1=conv2d_bn_(branch2_1x1,384,(3,1))
		branch2=concatenate([branch2_1x3,branch2_3x1])

		branch3_1x1=conv2d_bn_(x,448,(1,1))
		branch3_3x3=conv2d_bn_(branch3_1x1,384,(3,3))
		branch3_1x3=conv2d_bn_(branch3_3x3,384,(1,3))
		branch3_3x1=conv2d_bn_(branch3_3x3,384,(3,1))
		branch3=concatenate([branch3_1x3,branch3_3x1])

		branch4_pool=AveragePooling2D((3,3),strides=(1,1),padding="same")(x)
		branch4_pool=conv2d_bn_(branch4_pool,192,(1,1))

		x=concatenate([branch1_1x1,branch2,branch3,branch4_pool])
		return x

	input_=Input(shape=(ROW_SIZE,COL_SIZE,3))
	x=conv2d_bn_(input_,32,(3,3),strides=(2,2),padding="valid")
	x=conv2d_bn_(x,32,(3,3),padding="valid")
	x=conv2d_bn_(x,64,(3,3))
	x=MaxPooling2D((3,3),strides=(2,2))(x)

	x=conv2d_bn_(x,80,(1,1),padding="valid")
	x=conv2d_bn_(x,192,(3,3),padding="valid")
	x=MaxPooling2D((3,3),strides=(2,2))(x)

	#branch1
	x=block1(x)
	x=block1(x)

	#branch2
	branch1_3x3=conv2d_bn_(x,384,(3,3),strides=(2,2),padding="valid")

	branch2_1x1=conv2d_bn_(x,64,(1,1))
	branch2_3x3=conv2d_bn_(branch2_1x1,96,(3,3))
	branch2_3x3=conv2d_bn_(branch2_3x3,96,(3,3),strides=(2,2),padding="valid")

	branch3_pool=MaxPooling2D((3,3),strides=(2,2))(x)
	x=concatenate([branch1_3x3,branch2_3x3,branch3_pool])

	#branch3
	x=block2(x,128)
	x=block2(x,160)
	x=block2(x,160)
	x=block2(x,192)

	#branch4
	branch1_1x1=conv2d_bn_(x,192,(1,1))
	branch1_3x3=conv2d_bn_(branch1_1x1,320,(3,3),strides=(2,2),padding="valid")

	branch2_1x1=conv2d_bn_(x,192,(1,1))
	branch2_1x7=conv2d_bn_(branch2_1x1,192,(1,7))
	branch2_7x1=conv2d_bn_(branch2_1x7,192,(7,1))
	branch2_3x3=conv2d_bn_(branch2_7x1,192,(3,3),strides=(2,2),padding="valid")

	branch3_pool=MaxPooling2D((3,3),strides=(2,2))(x)

	x=concatenate([branch1_3x3,branch2_3x3,branch3_pool])

	#branch5
	x=block3(x)
	x=block3(x)



	model=Model(input_,x)
	#print(model.summary())
	return model

def vgg16():
	input_=Input(shape=(96,96,3))

	def conv_(filters,x):
		return Conv2D(int(filters*ALPHA),(3,3),padding="same",activation="relu")(x)

	#block1
	block1_conv1=conv_(64,input_)
	block1_conv2=conv_(64,block1_conv1)
	block1_pool1=MaxPooling2D()(block1_conv2)

	#block2
	block2_conv1=conv_(128,block1_pool1)
	block2_conv2=conv_(128,block2_conv1)
	block2_pool1=MaxPooling2D()(block2_conv2)

	#block3
	block3_conv1=conv_(256,block2_pool1)
	block3_conv2=conv_(256,block3_conv1)
	block3_conv3=conv_(256,block3_conv2)
	block3_pool1=MaxPooling2D()(block3_conv3)

	#block4
	block4_conv1=conv_(256,block3_pool1)
	block4_conv2=conv_(256,block4_conv1)
	block4_conv3=conv_(256,block4_conv2)
	block4_pool1=MaxPooling2D()(block4_conv3)

	#block5
	block5_conv1=conv_(256,block4_pool1)
	block5_conv2=conv_(256,block5_conv1)
	block5_conv3=conv_(256,block5_conv2)
	block5_pool1=MaxPooling2D()(block5_conv3)

	model=Model(input_,block5_pool1)

	return model

def mobilenetv2():
	def relu6(x):
		return K.relu(x,max_value=6)

	def conv_(inputs,filters,kernel_size=(3,3),strides=(1,1)):
		x=Conv2D(int(filters*ALPHA),kernel_size,strides=strides,padding="same")(inputs)
		x=BatchNormalization()(x)
		x=Activation(relu6)(x)
		return x

	def mini_bottleneck(inputs,filters,s,t,r=False):
		x=conv_(inputs,filters*t,kernel_size=(1,1))

		x=DepthwiseConv2D((3,3),strides=(s,s),depth_multiplier=1,padding="same")(x)
		x=BatchNormalization()(x)
		x=Activation(relu6)(x)

		x=Conv2D(int(filters*ALPHA),(1,1),strides=(1,1),padding="same")(x)
		x=BatchNormalization()(x)

		if r:
			x=add([x,inputs])

		return x

	def bottleneck(inputs,filters,n,s,t):
		x=mini_bottleneck(inputs,filters,s,t)

		for i in range(1,n):
			x=mini_bottleneck(x,filters,1,t,True)

		return x

	input_=Input(shape=(96,96,3))

	x=conv_(input_,32,strides=(2,2))

	x=bottleneck(x,16,1,1,1)
	x=bottleneck(x,24,2,2,6)
	x=bottleneck(x,32,3,2,6)
	x=bottleneck(x,64,4,2,6)
	x=bottleneck(x,96,3,1,6)
	x=bottleneck(x,160,3,2,6)
	x=bottleneck(x,320,1,1,6)

	model=Model(input_,x)

	return model

if __name__=="__main__":
	model=mobilenetv2()
	model.summary()
