import numpy as np
import cv2
import os

img_path='./test_img'
#sample_path='./pc_result'
def make_map(pred,img_h,img_w,img1_n,size):
	l=[]
	cd_map=np.zeros((img_h,img_w,3))
	i=0
	for p in pred:
		label=np.argmax(p)
		l.append(label)
	for h1 in range(0,img_h):
		for w1 in range(0,img_w):
			if int(h1-(size-1)/2)<0 or int(h1+(size-1)/2)>img_h-1 or int(w1-(size-1)/2)<0 or int(w1+(size-1)/2)>img_w-1:
				continue
			if l[i]==1:
				cd_map[h1,w1,0]=255
				cd_map[h1,w1,1]=255
				cd_map[h1,w1,2]=255
			i+=1
	img1=cv2.imread(os.path.join(img_path,img1_n))
	src_h=img1.shape[0]
	src_w=img1.shape[1]
	cd_map=cv2.resize(cd_map,(src_w,src_h))
	for h1 in range(0,src_h):
		for w1 in range(0,src_w): 
			if cd_map[h1,w1,0]>127:
				cd_map[h1,w1,0]=255
				cd_map[h1,w1,1]=255
				cd_map[h1,w1,2]=255
			else:
				cd_map[h1,w1,0]=0
				cd_map[h1,w1,1]=0
				cd_map[h1,w1,2]=0
	return cd_map
            
        
    
