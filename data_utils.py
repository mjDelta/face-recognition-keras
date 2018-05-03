import numpy as np
import os
from scipy.misc import imread
def triplet_generator(dir_path="lfw_alig2",batch_size=2):
	faces_dir=os.listdir(dir_path)
	np.random.seed(1)
	while True:
		anchor_faces=np.zeros((batch_size,96,96,3))
		posi_faces=np.zeros((batch_size,96,96,3))
		nega_faces=np.zeros((batch_size,96,96,3))

		for i in range(batch_size):	
			rand_idx=np.random.choice(len(faces_dir))
			anchor_dir=faces_dir[rand_idx]

			while len(os.listdir(os.path.join(dir_path,anchor_dir)))<=1:
				rand_idx=np.random.choice(len(faces_dir))
				anchor_dir=faces_dir[rand_idx]	
			
			anchor_face_idx=np.random.choice(len(os.listdir(os.path.join(dir_path,anchor_dir))))
			posi_face_idx=np.random.choice(len(os.listdir(os.path.join(dir_path,anchor_dir))))
			while anchor_face_idx==posi_face_idx:
				posi_face_idx=np.random.choice(len(os.listdir(os.path.join(dir_path,anchor_dir))))

			anchor_face=os.path.join(dir_path,anchor_dir,os.listdir(os.path.join(dir_path,anchor_dir))[anchor_face_idx])
			posi_face=os.path.join(dir_path,anchor_dir,os.listdir(os.path.join(dir_path,anchor_dir))[posi_face_idx])

			neg_idx=np.random.choice(len(faces_dir))
			while neg_idx==rand_idx:
				neg_idx=np.random.choice(len(faces_dir))
			neg_dir=faces_dir[neg_idx]

			neg_face=os.path.join(dir_path,neg_dir,os.listdir(os.path.join(dir_path,neg_dir))[np.random.choice(len(os.listdir(os.path.join(dir_path,neg_dir))))])

			anchor=imread(anchor_face)/255.
			posi=imread(posi_face)/255.
			nega=imread(neg_face)/255.

			anchor_faces[i]=anchor
			posi_faces[i]=posi
			nega_faces[i]=nega
			'''if i%5==0:
				print("anchor: ",anchor_face)
				print("positive: ",posi_face)
				print("negative: ",neg_face)'''

		yield [anchor_faces,posi_faces,nega_faces],None		
if __name__=="__main__":
	g=triplet_generator()
	for i in range(10):
		inputs,_=g.__next__()
		print("="*30)
