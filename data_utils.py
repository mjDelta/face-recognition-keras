import numpy as np
import os
from scipy.misc import imread
from sklearn.metrics.pairwise import pairwise_distances
import itertools
import gc
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

class triplet_generator2:
	def __init__(self,model,dir_path="lfw_alig2/",batch_size=32,alpha=0.25):
		self.model=model
		self.dir_path=dir_path
		self.batch_size=batch_size
		self.alpha=alpha

	def random_choose_faces(self):
		faces_dirs=os.listdir(self.dir_path)
		faces_paths=[os.path.join(self.dir_path,dir_) for dir_ in faces_dirs]

		choosed_faces_dirs=np.random.choice(faces_paths,self.batch_size,replace=False)

		choosed_faces_values=[imread(os.path.join(face_dir,face_name))/255. for face_dir in choosed_faces_dirs for face_name in os.listdir(face_dir)]
		choosed_faces_names=[face_dir[len(self.dir_path):] for face_dir in choosed_faces_dirs for face_name in os.listdir(face_dir)]

		return choosed_faces_values,choosed_faces_names

	def get_triplet(self,choosed_faces_values,choosed_faces_names):
		anchors=[];positives=[];negatives=[]
		anchors_names=[];positives_names=[];negatives_names=[]
		choosed_faces_values=np.array(choosed_faces_values)
		choosed_faces_values=choosed_faces_values.reshape((len(choosed_faces_values),96,96,3))
		choosed_faces_embeddings=self.model.predict(choosed_faces_values)
		
		dist_mat=pairwise_distances(choosed_faces_embeddings,metric="sqeuclidean")
		cats=np.unique(choosed_faces_names,axis=0)
		for c in cats:
			pos_samples=np.where(c==np.array(choosed_faces_names))[0]
			if len(pos_samples)==1:
				continue
			for i,j in itertools.combinations(pos_samples,2):
				pos_dist=dist_mat[i,j]
				neg_conds=np.where((c!=np.array(choosed_faces_names))*(dist_mat[i]-pos_dist<self.alpha))
				if len(neg_conds)>0:
					rand_idx=np.random.choice(len(neg_conds))
					idx=neg_conds[rand_idx]
					anchors.append(choosed_faces_values[i]);positives.append(choosed_faces_values[j]);negatives.append(choosed_faces_values[idx])
					anchors_names.append(choosed_faces_names[i]);positives_names.append(choosed_faces_names[j]);negatives_names.append(choosed_faces_names[idx])
		print("get_triplets"+str(len(anchors)))
		del choosed_faces_values
		del choosed_faces_names
		gc.collect()
		return anchors,positives,negatives,anchors_names,positives_names,negatives_names
	def flow(self):
		while True:	
			anchors=[];positives=[];negatives=[]
			values,names=self.random_choose_faces()
			ans,pos,nes,anchors_names,positives_names,negatives_names=self.get_triplet(values,names)
			anchors.extend(ans);positives.extend(pos);negatives.extend(nes)
			del ans
			del pos
			del nes
			gc.collect()
			while(len(anchors)<self.batch_size):
				values,names=self.random_choose_faces()
				ans,pos,nes,anchors_names,positives_names,negatives_names=self.get_triplet(values,names)
				anchors.extend(ans);positives.extend(pos);negatives.extend(nes)
				del ans
				del pos
				del nes
				gc.collect()
			anchors=np.array(anchors[:self.batch_size]);positives=np.array(positives[:self.batch_size]);negatives=np.array(negatives[:self.batch_size])
			yield [anchors,positives,negatives],None

if __name__=="__main__":
	'''g=triplet_generator()
	for i in range(10):
		inputs,_=g.next()
		print(inputs[0][0,0,0,0])
		print("="*30)'''
	triplet_generator2()
