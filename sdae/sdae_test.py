from sdae import sdae, load_data, make_map
#from sdae import StackedDenoisingAE
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
#from load_data_2 import load_data
import cv2
#from make_map import make_map
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler
from collections import Counter
from keras.models import load_model
np.set_printoptions(threshold=np.inf)  

n_classes = 2

#test image pair
img_path='./test_img'
    
def classification(img1_n,img2_n,sample_img):
	img1=cv2.imread(os.path.join(img_path,img1_n),-1)
	print('shape of src img1: '+str(img1.shape))
	if 'weihe' in img1_n:
		img1=cv2.resize(img1,(int(img1.shape[1]*0.25),int(img1.shape[0]*0.25)))
	else:
		img1=cv2.resize(img1,(int(img1.shape[1]*0.5),int(img1.shape[0]*0.5)))
	print('shape of resized img1: '+str(img1.shape))
	img_h=img1.shape[0]
	img_w=img1.shape[1]
	print('preparing the training, testing, and predicting data')
	(X_train_1, Y_train_1), (X_test_1, Y_test_1) ,(X_train_2, Y_train_2), (X_test_2, Y_test_2),X_PRE,size= load_data.load_data(img1_n,img2_n,sample_img)  
    
	Y_train_1=np.array(Y_train_1,'float32')
	Y_test_1=np.array(Y_test_1,'float32')
	Y_train_1 /= 255#归一化
	Y_test_1 /= 255

	n_in = n_out = Y_train_1.shape[1];#输入层的神经元个数
	cur_sdae = sdae.StackedDenoisingAE(n_layers = 5, n_hid = [500,200,100,50,20], dropout = [0.1], nb_epoch = 10)#define sdae network

	#train a stacked denoising autoencoder and get the trained model, dense representations of the final hidden layer, and reconstruction error
	model, (dense_train, dense_val, dense_test), recon_mse = cur_sdae.get_pretrained_sda(Y_train_1, Y_test_1, Y_test_1, './sdae/output/',img1_n)
    
	counter=sorted(Counter(Y_train_2).items())
	neg=counter[0][1]
	pos=counter[1][1]
	print('negative samples before :'+' '+str(neg))
	print('positive samples before :'+' '+str(pos))
	if len(counter)!=1:  
		rs=RandomUnderSampler(random_state=0,replacement=True)
		X_train_2, Y_train_2 = rs.fit_sample(X_train_2, Y_train_2)
		counter=sorted(Counter(Y_train_2).items())
		neg=counter[0][1]
		pos=counter[1][1]
		print('negative samples :'+' '+str(neg))
		print('positive samples :'+' '+str(pos))
	else:
		pass
    
	np.random.seed(116)
	np.random.shuffle(X_train_2) 
	np.random.seed(116)
	np.random.shuffle(Y_train_2) 
	X_train_2, X_val_2,Y_train_2, Y_val_2 = train_test_split( X_train_2, Y_train_2, test_size=0.2, random_state=1000,shuffle=True)

	X_train_2=np.array(X_train_2,'float32')
	X_test_2=np.array(X_test_2,'float32')
	X_val_2=np.array(X_val_2,'float32')
	Y_train_2=np.array(Y_train_2)
	Y_test_2=np.array(Y_test_2)
	Y_val_2=np.array(Y_val_2)
	X_train_2 /= 255#归一化
	X_test_2 /= 255
	X_val_2 /= 255

	# convert class vectors to binary class matrices
	Y_train_2 = to_categorical(Y_train_2, n_classes)#one hot
	Y_test_2 = to_categorical(Y_test_2, n_classes)
	Y_val_2 = to_categorical(Y_val_2, n_classes)

	fit_classifier,_tuple,_error = cur_sdae.supervised_classification(model=model, x_train=X_train_2, x_val=X_val_2, y_train=Y_train_2, y_val=Y_val_2,x_test=X_test_2,y_test=Y_test_2, n_classes=n_classes,dir1=img1_n)
	# load weights 加载模型权重
	print('loading model...')
	fit_classifier.load_weights('./sdae/weight/'+str(img1_n)+'weights-improvement.hdf5')
	X_PRE=np.array(X_PRE,'float32')
	X_PRE /= 255
	print('predicting...')
	pred = cur_sdae.predict(fit_classifier, X_PRE)
	print('making change detection map...')
	cd_map=make_map.make_map(pred,img_h,img_w,img1_n,size)
	#cv2.imwrite(os.path.join('./cd_result',img1_n),cd_map,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	return cd_map
if __name__=='__main__':
	cd_map=classification(img1_n,img2_n,sample_img)
    
