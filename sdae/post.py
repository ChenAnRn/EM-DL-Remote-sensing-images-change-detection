import cv2
import os
import numpy as np

gt_path='./gt'

def post_process(cd_map,img1_n):
	im=img1_n[:-5]+'_gt.png'
	img_gt=cv2.imread(os.path.join(gt_path,im))
	shape=img_gt.shape
	cd_gray = cv2.cvtColor(cd_map.astype(np.uint8),cv2.COLOR_RGB2GRAY)
	contours,hierarch=cv2.findContours(cd_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)    
	for i in range(len(contours)):        
		area = cv2.contourArea(contours[i])        
		if area < 100:            
			cv2.drawContours(cd_gray,[contours[i]],0,0,-1)
	cv2.imwrite(os.path.join('./cd_result',img1_n[:-5]+'.png'),cd_gray,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
	print('obtaining the final change detection map!')
	FN=0
	FP=0
	TN=0
	TP=0
	for a in range(0,shape[0]):
		for b in range(0,shape[1]):
			x=cd_gray[a,b]
			y=img_gt[a,b,0]
			if x==0 and y==255:
				FN+=1
			if x==255 and y==0:
				FP+=1
			if x==255 and y==255:
				TP+=1
			if x==0 and y==0:
				TN+=1  
	n=FN+FP+TN+TP
	print(n)
	CA=(TP+TN)/n
	FA=FP/n
	MA=FN/n
	PRE=(TP+FP)*(TP+FN)/(n*n)+(TN+FP)*(FN+TN)/(n*n)
	KC=(CA-PRE)/(1-PRE) 
	OE=(FP+FN)/n
	print('KC:  '+str(KC))
	print('FA:  '+str(FA))
	print('MA:  '+str(MA))
	print('OE:  '+str(OE))
	print('CA:  '+str(CA))
