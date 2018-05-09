import numpy as np
import os
from scipy.misc import imread
from sklearn.metrics.pairwise import pairwise_distances
import itertools

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

def triplet_generator2(model,dir_path="lfw_alig2/",batch_size=32):
	def random_choose_faces(dir_path,batch_size):
		faces_dirs=os.listdir(dir_path)
		faces_paths=[os.path.join(dir_path,dir_) for dir_ in faces_dirs]

		choosed_faces_dirs=np.random.choice(faces_paths,batch_size,replace=False)

		choosed_faces_values=[imread(os.path.join(face_dir,face_name))/255. for face_dir in choosed_faces_dirs for face_name in os.listdir(face_dir)]
		choosed_faces_names=[face_dir[len(dir_path):] for face_dir in choosed_faces_dirs for face_name in os.listdir(face_dir)]

		return choosed_faces_values,choosed_faces_names

	def get_triplet(model,choosed_faces_values,choosed_faces_names,alpha=0.25):
		anchors=[];positives=[];negatives=[]
		anchors_names=[];positives_names=[];negatives_names=[]
		choosed_faces_values=np.array(choosed_faces_values)
		choosed_faces_values=choosed_faces_values.reshape((len(choosed_faces_values),96,96,3))
		choosed_faces_embeddings=model.predict(choosed_faces_values)
		
		dist_mat=pairwise_distances(choosed_faces_embeddings,metric="sqeuclidean")
		cats=np.unique(choosed_faces_names,axis=0)
		for c in cats:
			pos_samples=np.where(c==np.array(choosed_faces_names))[0]
			if len(pos_samples)==1:
				continue
			for i,j in itertools.combinations(pos_samples,2):
				pos_dist=dist_mat[i,j]
				neg_conds=np.where((c!=np.array(choosed_faces_names))*(dist_mat[i]-pos_dist<alpha))
				if len(neg_conds)>0:
					rand_idx=np.random.choice(len(neg_conds))
					idx=neg_conds[rand_idx]
					anchors.append(choosed_faces_values[i]);positives.append(choosed_faces_values[j]);negatives.append(choosed_faces_values[idx])
					anchors_names.append(choosed_faces_names[i]);positives_names.append(choosed_faces_names[j]);negatives_names.append(choosed_faces_names[idx])
		print("get_triplets"+str(len(anchors)))
		return anchors,positives,negatives,anchors_names,positives_names,negatives_names

	while True:	
		anchors=[];positives=[];negatives=[]
		values,names=random_choose_faces(dir_path,batch_size)
		ans,pos,nes,anchors_names,positives_names,negatives_names=get_triplet(model,values,names)
		anchors.extend(ans);positives.extend(pos);negatives.extend(nes)
		while(len(anchors)<batch_size):
			values,names=random_choose_faces(dir_path,batch_size)
			ans,pos,nes,anchors_names,positives_names,negatives_names=get_triplet(model,values,names)
			anchors.extend(ans);positives.extend(pos);negatives.extend(nes)
		anchors=np.array(anchors[:batch_size]);positives=np.array(positives[:batch_size]);negatives=np.array(negatives[:batch_size])
		yield [anchors,positives,negatives],None



			
if __name__=="__main__":
	'''g=triplet_generator()
	for i in range(10):
		inputs,_=g.next()
		print(inputs[0][0,0,0,0])
		print("="*30)'''
	triplet_generator2()
