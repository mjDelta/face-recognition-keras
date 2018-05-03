# face-recognition-keras
The procedure of this repository includes `face detection`, `affine transformation`, `extract face features`, `find a threshold to spilt faces`. And the results are evaluated on the [LFW](http://vis-www.cs.umass.edu/lfw/) datasets.</br>

### Requirements:</br>
* dlib(19.10.0)
* keras(2.1.6)
* tensorflow(1.7.0) 
* opencv-python(3.4.0.12)

## Todo List
- [x] InceptionV3 backend
- [ ] MobileNet backend
- [ ] VGG16 backend
- [ ] ResNet50 backend
- [ ] Xception backend
- [ ] DenseNet backend

## Face Detection and Affine Transformation
I use Dlib and opencv for this preprocessing procedure <a href="https://github.com/mjDelta/face-recognition-keras/blob/master/align_face.py">align_face.py</a>. Dlib does the quick face detection, while opencv does cropping and affine transformation. </br>
![image](https://github.com/mjDelta/face-recognition-keras/blob/master/imgs/preprocessing.png)</br>
## Features Extraction with Deep Learning
I use several basical deep learning model to extract 128 features from the preprocessed images. And the loss is `triplet loss`, which I think is the core of facenet. Triplet loss convert the distance among those embeddings into Euclidean distance and do BP operator on the triplet loss to optimize feature extractor.</br>
In <a href="https://github.com/mjDelta/face-recognition-keras/blob/master/backends.py">backends.py</a>, I defined sever base deep learning model as the base of feature extractor, making it easier to switch different bases and compare results.</br>
In <a href="https://github.com/mjDelta/face-recognition-keras/blob/master/frontends.py">frontends.py</a>, I combined the backends with `Dense` Layer as the final feature extractor and defined the triplet loss.</br>
In <a href="https://github.com/mjDelta/face-recognition-keras/blob/master/train.py">train.py</a>, I trained models with feature extractor.</br>
## Choose a Threshold
As triplet loss is defined by Euclidean distance, we don't have a threshold to split embeddings. So, we need to choose a threshold.</br>
In <a href="https://github.com/mjDelta/face-recognition-keras/blob/master/test.py">test.py</a>, I searched threshold violently.</br>
## Results
<br>
		<table style='border:1px solid #e8e8e8;'>
		<thead>
			<tr>
				<th></th>
				<th>INCEPTIONV3</th>
				<th>VGG16</th>
        <th>MOBILENET</th>
        <th>RESNET50</th>
        <th>XCEPTION</th>
        <th>DENSENET</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>BEST THRESHOLD</th>
				<td>0.1</td>
				<td></td>
        <td></td>
				<td></td>
        <td></td>
				<td></td>
			</tr>
			<tr>
				<th>ACC</th>
				<td>96.86%</td>
				<td></td>
        <td></td>
				<td></td>
        <td></td>
				<td></td>
			</tr>
			<tr>
				<th>F-MEASURE</th>
				<td>87.5%</td>
				<td></td>
        <td></td>
				<td></td>
        <td></td>
				<td></td>
			</tr>
		</tbody>
		</table>
