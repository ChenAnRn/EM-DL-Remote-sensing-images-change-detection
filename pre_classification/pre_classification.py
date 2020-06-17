import numpy as np
import cv2
import os

img_path='./test_img'#try deleting to check global variable
edge_path='./pre_classification/image-edge'
#global pc_map

	#---------------pre-classification parameters--------------------
size=7
mean_threshold=0.1
var_threshold=0.01
    
def overlap_img(img1,img2):
	shape=img1.shape
	overlap=np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
	for i in range(shape[0]):
		for j in range(shape[1]):
			x=img1[i,j]
			y=img2[i,j]
			if x==255 and y==255:
				overlap[i,j,0]=255
				overlap[i,j,1]=255
				overlap[i,j,2]=255
			elif x==0 and y==0:
				overlap[i,j,0]=0
				overlap[i,j,1]=0
				overlap[i,j,2]=0
			elif x==0 and y==255:#the edge of img1 is red
				overlap[i,j,0]=0
				overlap[i,j,1]=0
				overlap[i,j,2]=255
			elif x==255 and y==0:#the edge of img2 is green
				overlap[i,j,0]=0
				overlap[i,j,1]=255
				overlap[i,j,2]=0
	return overlap
   
            
    
def edge_binarize(img1_n,img2_n):#maybe some errors because of parameters
	print('threshold processing...')
	img1_simple_name=img1_n[:-4]+'_simple.png'
	img1_adap_name=img1_n[:-4]+'_adaptive.png'
	img2_simple_name=img2_n[:-4]+'_simple.png'
	img2_adap_name=img2_n[:-4]+'_adaptive.png'
	img1_edge_name=img1_n[:-4]+'_binary.png'
	img2_edge_name=img2_n[:-4]+'_binary.png'
	img1=cv2.imread(os.path.join(edge_path,img1_n),0)
	img2=cv2.imread(os.path.join(edge_path,img2_n),0)
	#print(type(img1))
	_, img1_simple = cv2.threshold(img1,90, 255, cv2.THRESH_BINARY)
	img1_adap = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,3)
	_, img2_simple = cv2.threshold(img2,90, 255, cv2.THRESH_BINARY)
	img2_adap = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,3)
	#save the result
	cv2.imwrite(os.path.join('./pre_classification/threshold_result/',img1_simple_name),img1_simple,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	cv2.imwrite(os.path.join('./pre_classification/threshold_result/',img1_adap_name),img1_adap,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	cv2.imwrite(os.path.join('./pre_classification/threshold_result/',img2_simple_name),img2_simple,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	cv2.imwrite(os.path.join('./pre_classification/threshold_result/',img2_adap_name),img2_adap,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	print('save the result of threshold processing!')
	print('combining two threshold processing result...')
	shape=img1.shape
	for a in range(0,shape[0]):
		for b in range(0,shape[1]):
			if img1_simple[a,b]==255:#maybe some errors because of dims
				img1_adap[a,b]=255  
	for a in range(0,shape[0]):
		for b in range(0,shape[1]):
			if img2_simple[a,b]==255:#maybe some errors because of dims
				img2_adap[a,b]=255 
	cv2.imwrite(os.path.join('./pre_classification/binary_result',img1_edge_name),img1_adap,[int(cv2.IMWRITE_PNG_COMPRESSION),0]) 
	cv2.imwrite(os.path.join('./pre_classification/binary_result',img2_edge_name),img2_adap,[int(cv2.IMWRITE_PNG_COMPRESSION),0]) 
	print('save the binary edge map!')
	return img1_adap,img2_adap

def analyze(h1,w1,t,img,img1,img2,h,w):
	block_=img[h1:h1+size,w1:w1+size,:]
	block_gray = cv2.cvtColor(block_,cv2.COLOR_RGB2GRAY)
	_, block_gray = cv2.threshold(block_gray,250, 255,cv2.THRESH_BINARY_INV)
	contour, hier = cv2.findContours(block_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for con in contour:
		for c in con:
			search_1(h1+c[0][0],w1+c[0][1],img,img1,img2,h,w)
	while t>0:
		for m in range(h1,size+h1):
			for n in range(w1,size+w1):
				search_2(m,n,pc_map,img1,img2,h,w)
		t-=1

def search_1(i,j,image,img1,img2,h,w):
	n=np.zeros(9)
	if i-1<0 or j-1<0 or i+1>h-1 or j+1>w-1: 
		pass
	elif image[i,j,0]==0: 
		center=np.sqrt(np.sum(np.square(img1[i,j]-img2[i,j])))    
		n[0]=np.sqrt(np.sum(np.square(img1[i-1,j]-img2[i-1,j]))) #left
		n[1]=np.sqrt(np.sum(np.square(img1[i+1,j]-img2[i+1,j]))) #right
		n[2]=np.sqrt(np.sum(np.square(img1[i,j-1]-img2[i,j-1]))) #up
		n[3]=np.sqrt(np.sum(np.square(img1[i,j+1]-img2[i,j+1]))) #down
		n[4]=np.sqrt(np.sum(np.square(img1[i-1,j-1]-img2[i-1,j-1]))) #up  left
		n[5]=np.sqrt(np.sum(np.square(img1[i+1,j-1]-img2[i+1,j-1]))) #up  right
		n[6]=np.sqrt(np.sum(np.square(img1[i-1,j+1]-img2[i-1,j+1]))) #down  left
		n[7]=np.sqrt(np.sum(np.square(img1[i+1,j+1]-img2[i+1,j+1]))) #down  right
		n[8]=center
		mean=np.mean(n[:-1])
		var=np.var(n)
		if (np.abs(mean-center))>mean_threshold or var>var_threshold:   #changed
		#if (np.abs(mean-center))>0.05 or var>var_threshold:
			pc_map[i-1:i+1,j-1:j+1,0]=255
			pc_map[i-1:i+1,j-1:j+1,1]=255
			pc_map[i-1:i+1,j-1:j+1,2]=255

def search_2(i,j,image,img1,img2,h,w):
	n=np.zeros(9)
	if i-1<0 or j-1<0 or i+1>h-1 or j+1>w-1: 
		pass
	elif image[i,j,0]==255: 
		center=np.sqrt(np.sum(np.square(img1[i,j]-img2[i,j])))     
		n[0]=np.sqrt(np.sum(np.square(img1[i-1,j]-img2[i-1,j]))) #left
		n[1]=np.sqrt(np.sum(np.square(img1[i+1,j]-img2[i+1,j]))) #right
		n[2]=np.sqrt(np.sum(np.square(img1[i,j-1]-img2[i,j-1]))) #up
		n[3]=np.sqrt(np.sum(np.square(img1[i,j+1]-img2[i,j+1]))) #down
		n[4]=np.sqrt(np.sum(np.square(img1[i-1,j-1]-img2[i-1,j-1]))) #up  left
		n[5]=np.sqrt(np.sum(np.square(img1[i+1,j-1]-img2[i+1,j-1]))) #up  right
		n[6]=np.sqrt(np.sum(np.square(img1[i-1,j+1]-img2[i-1,j+1]))) #down  left
		n[7]=np.sqrt(np.sum(np.square(img1[i+1,j+1]-img2[i+1,j+1]))) #down  right
		n[8]=center
		mean=np.mean(n[:-1])
		var=np.var(n)
		if (np.abs(mean-center))<mean_threshold and var<var_threshold:   #changed
			pc_map[i-1:i+1,j-1:j+1,0]=255
			pc_map[i-1:i+1,j-1:j+1,1]=255
			pc_map[i-1:i+1,j-1:j+1,2]=255

def count(block,size):
	nc=0
	n_black=0
	for m in range(0,size):
		for n in range(0,size):
			if block[m,n,2]==255 and block[m,n,1]==0 and block[m,n,0]==0:
				nc+=1
			elif block[m,n,2]==0 and block[m,n,1]==255 and block[m,n,0]==0:
				nc+=1
			elif block[m,n,2]==0 and block[m,n,1]==0 and block[m,n,0]==0:
				n_black+=1
	return nc,n_black
	
def pre_classification(img1_n,img2_n,img1_binary,img2_binary):
	print('overlapping two binary edge map...')
	overlap=np.zeros((img1_binary.shape[0],img1_binary.shape[1],3),dtype=np.uint8)
	overlap=overlap_img(img1_binary,img2_binary)
	overlap_name=img1_n[:-5]+'_overlap.png'
	cv2.imwrite(os.path.join('./pre_classification/overlap_result/',overlap_name),overlap,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	print('save the result of overlapped edge map!')

	print('executing the pre-classification algorithm...')

	img_pc_name=img1_n[:-5]+'_pc.png'
	img1=cv2.imread(os.path.join(img_path,img1_n),-1)
	img2=cv2.imread(os.path.join(img_path,img2_n),-1)
	h,w,_=img1.shape
	global pc_map
	pc_map=np.zeros((h,w,3),dtype=np.uint8)
	block=np.ones((size,size,3),dtype=np.uint8)
	img1=img1/255
	img2=img2/255
	for h1 in range(0,h,5):
		for w1 in range(0,w,5):
			if h1+size>h or w1+size>w:
				continue
			else:
				block=overlap[h1:h1+size,w1:w1+size,:]
				nc,n_black=count(block,size)    #red and green and black
				nc=nc+n_black
				if nc==0:    #unchanged
					pass
				elif nc<size*size:#search points
					analyze(h1,w1,5,overlap,img1,img2,h,w)
	cv2.imwrite(os.path.join('./pre_classification/pc_result/',img_pc_name),pc_map,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	print('save the pre-classification result!')
	return pc_map

if __name__=='__main__':
	img1_binary,img2_binary=edge_binarize(img1,img2)
	pc_map=pre_classification(img1,img2,img1_binary,img2_binary)
\
