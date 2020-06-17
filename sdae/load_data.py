import numpy as np
import os 
from sklearn.model_selection import train_test_split
import cv2
import skimage

img_path='./test_img'

def load_data(img1_n,img2_n,sample_img):
	size=5
	square=size*size
	block_1=np.ones((size,size,3))
	block_2=np.ones((size,size,3))
	block_3=np.ones((size,size,3))
	block_4=np.ones((size,size,3))
	X_1=[]#train set used to pretrain sdae
	X_2=[]#train set used to fine-tune sdae
	y_1=[]
	y_2=[]
	X_PRE=[]
	#-------------------------------------------X_train_1------------------------------------------#
	img1=cv2.imread(os.path.join(img_path,img1_n),-1)
	img2=cv2.imread(os.path.join(img_path,img2_n),-1)
	if 'weihe' in img1_n:
		img1=cv2.resize(img1,(int(img1.shape[1]*0.25),int(img1.shape[0]*0.25)))
		img2=cv2.resize(img2,(int(img2.shape[1]*0.25),int(img2.shape[0]*0.25)))
	else:
		img1=cv2.resize(img1,(int(img1.shape[1]*0.5),int(img1.shape[0]*0.5)))
		img2=cv2.resize(img2,(int(img2.shape[1]*0.5),int(img2.shape[0]*0.5)))
	img1=cv2.cvtColor(img1,cv2.COLOR_RGBA2RGB)
	img1_noise = skimage.util.random_noise(img1,mode='gaussian',seed=None,clip=True)
	h,w,_=img1.shape
	
	img2=cv2.cvtColor(img2,cv2.COLOR_RGBA2RGB)
	img2_noise = skimage.util.random_noise(img2,mode='gaussian',seed=None,clip=True)
	print('shape of training img1: '+str(img1.shape))
	print('shape of training img2: '+str(img2.shape))
	for h1 in range(0,h,size):
		for w1 in range(0,w,size):
			if h1+size>h-1 or w1+size>w-1:
				continue
			else:
				block_1=img1_noise[h1:h1+size,w1:w1+size,:]  
				block_2=img2_noise[h1:h1+size,w1:w1+size,:]  
				block_3=img1[h1:h1+size,w1:w1+size,:]  
				block_4=img2[h1:h1+size,w1:w1+size,:]

				vector_1=block_1.flatten()
				vector_2=block_2.flatten()
				vector_3=block_3.flatten()
				vector_4=block_4.flatten()
				vector_X=vector_1+vector_2
				vector_Y=vector_3+vector_4
				X_1.append(vector_X)
				y_1.append(vector_Y)

	X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split( X_1, y_1, test_size=0.2, random_state=11)

	#-------------------------------------------X_train_2------------------------------------------#     
	#变化部分    
	sample_img=cv2.resize(sample_img,(img1.shape[1],img1.shape[0]))
	print('sample_img.shape'+str(sample_img.shape))
	for h1 in range(0,h):
		for w1 in range(0,w): 
			if sample_img[h1,w1,0]>200:
				sample_img[h1,w1,0]=255
				sample_img[h1,w1,1]=255
				sample_img[h1,w1,2]=255
			elif sample_img[h1,w1,0]<100:
				sample_img[h1,w1,0]=0
				sample_img[h1,w1,1]=0
				sample_img[h1,w1,2]=0
			else:
				sample_img[h1,w1,0]=127
				sample_img[h1,w1,1]=127
				sample_img[h1,w1,2]=127
	for h1 in range(0,h):
		for w1 in range(0,w):
			if int(h1-(size-1)/2)<0 or int(h1+(size-1)/2)>h-1 or int(w1-(size-1)/2)<0 or int(w1+(size-1)/2)>w-1:
				continue
			elif sample_img[h1,w1,0]==255:
				block_1=img1[int(h1-(size-1)/2):int(h1+(size-1)/2+1),int(w1-(size-1)/2):int(w1+(size-1)/2+1),:]
				block_2=img2[int(h1-(size-1)/2):int(h1+(size-1)/2+1),int(w1-(size-1)/2):int(w1+(size-1)/2+1),:]
				vector_1=block_1.flatten()
				vector_2=block_2.flatten()
				vector_X=vector_1+vector_2
				X_2.append(vector_X)
				y_2.append(1)
				
	#未变化部分    
	for h1 in range(0,h):
		for w1 in range(0,w):
			if int(h1-(size-1)/2)<0 or int(h1+(size-1)/2)>h-1 or int(w1-(size-1)/2)<0 or int(w1+(size-1)/2)>w-1:
				continue
			elif sample_img[h1,w1,0]==0:
				block_1=img1[int(h1-(size-1)/2):int(h1+(size-1)/2+1),int(w1-(size-1)/2):int(w1+(size-1)/2+1),:]
				block_2=img2[int(h1-(size-1)/2):int(h1+(size-1)/2+1),int(w1-(size-1)/2):int(w1+(size-1)/2+1),:]
				vector_1=block_1.flatten()
				vector_2=block_2.flatten()
				vector_X=vector_1+vector_2
				X_2.append(vector_X)
				y_2.append(0)

	X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split( X_2, y_2, test_size=0.2, random_state=11)

	for h1 in range(0,h):
		for w1 in range(0,w):
			if int(h1-(size-1)/2)<0 or int(h1+(size-1)/2)>h-1 or int(w1-(size-1)/2)<0 or int(w1+(size-1)/2)>w-1:
				continue
			else:
				block_1=img1[int(h1-(size-1)/2):int(h1+(size-1)/2+1),int(w1-(size-1)/2):int(w1+(size-1)/2+1),:]
				block_2=img2[int(h1-(size-1)/2):int(h1+(size-1)/2+1),int(w1-(size-1)/2):int(w1+(size-1)/2+1),:]
			vector_1=block_1.flatten()
			vector_2=block_2.flatten()
			vector_X=vector_1+vector_2
			X_PRE.append(vector_X)

	return  (X_train_1, y_train_1), (X_test_1,y_test_1),(X_train_2, y_train_2),( X_test_2, y_test_2),X_PRE,size
    
    
