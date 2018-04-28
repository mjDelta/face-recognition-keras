import dlib 
from skimage.io import imread,imsave
import cv2
import numpy as np
from matplotlib import pyplot as plt
OUTER_EYES_AND_NOSE = [36, 45, 33]
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

def get_bounding_box(img):
	detector=dlib.get_frontal_face_detector()
	

	dets=detector(img)
	if len(dets)>1 or len(dets)==0:
		return -1
	
	return dets[0]

def get_face_shape(img,bb):##return 68 keypoints of face
	model_path="models/landmarks.dat"
	predictor=dlib.shape_predictor(model_path)
	pnts=predictor(img,bb)
	return np.float32(list(map(lambda p:(p.x,p.y),pnts.parts())))

def align_face(img_path,size):
	img=imread(img_path)
	bb=get_bounding_box(img)
	if np.sum(bb)==-1:
		return -1
	landmarks=get_face_shape(img,bb)

	landmarksindices=np.array(OUTER_EYES_AND_NOSE)
	
	H=cv2.getAffineTransform(landmarks[landmarksindices],\
													size*MINMAX_TEMPLATE[landmarksindices])

	thumbnail=cv2.warpAffine(img,H,(size,size))
	'''
	img_copy=img.copy()
	cv2.rectangle(img_copy,(bb.left(),bb.top()),(bb.right(),bb.bottom()),(255,0,0),2)
	fig=plt.figure(figsize=(6,2))
	fig.add_subplot(131)
	plt.imshow(img)
	fig.add_subplot(132)
	plt.imshow(img_copy)
	fig.add_subplot(133)
	plt.imshow(thumbnail)
	plt.savefig("preprocessing.png")
	plt.show()'''
	return thumbnail

if __name__=="__main__":
	import os
	dir_="lfw/"
	out_dir_="lfw_alig/"
	count=0
	for d in os.listdir(dir_):
		for f in os.listdir(os.path.join(dir_,d)):
			if count%50==0:
				print(count)
			img_path=os.path.join(dir_,d,f)
			size=96
			img=align_face(img_path,size)
			if np.sum(img)==-1:
				continue
			out_dir=os.path.join(out_dir_,d)
			out_path=os.path.join(out_dir_,d,f)
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			imsave(out_path,img)
			count+=1
			
		
