from sample_selection import slic_img
import numpy as np
import os 
import cv2

img_path='./test_img'
pc_path='./pre_classification/pc_result'

#pc_img=cv2.imread(os.path.join(pc_path,'farmland_test.png'),-1)
#sample_img=np.copy(pc_img)


def make_map(img,p,super1):
	for s in super1:
		img[s[0],s[1],0]=p
		img[s[0],s[1],1]=p
		img[s[0],s[1],2]=p


def analyze(super1,pc_img,sample_img):
	white=0
	black=0
	grey=0
	for s in super1:
		pixel=pc_img[s[0],s[1],0]
		if pixel==255:
			white+=1
		elif pixel==0:
			black+=1
		else:
			grey+=1
	if white>len(super1)*0.8:
		pass
	elif white>0:
		make_map(sample_img,127,super1)
		
def analyze2(super2,pc_img,sample_img):
	white=0
	black=0
	grey=0
	for s in super2:
		pixel=sample_img[s[0],s[1],0]
		if pixel==255:
			white+=1
		elif pixel==0:
			black+=1
		else:
			grey+=1
	if black==len(super2):
		pass
	elif black>0:
		for s in super2:
			if pc_img[s[0],s[1],0]==0:
				sample_img[s[0],s[1],0]=127
				sample_img[s[0],s[1],1]=127
				sample_img[s[0],s[1],2]=127

def super_segmentation(img1_n,img2_n):
	print('superpixel segmentation with small scale ')
	print('superpixel segmentation with large scale ')
	if 'farmland' in img1_n:
		p = slic_img.SLICProcessor(os.path.join(img_path,img1_n), 5000, 20)
	if 'forest' in img1_n:
		p = slic_img.SLICProcessor(os.path.join(img_path,img1_n), 9000, 20)    
	if 'weihe' in img1_n:
		p = slic_img.SLICProcessor(os.path.join(img_path,img1_n), 100000, 20)
	p.iterate_10times(img1_n)

def sample_selection(img1_n,img2_n,sample_img):
	print('selecting samples...')
	inte_superpixel1=np.load(os.path.join('./sample_selection/img_contour/7',img1_n[:-4]+'_super.npy'))
	inte_superpixel2=np.load(os.path.join('./sample_selection/img_contour/2',img1_n[:-4]+'_super.npy'))
	inte_superpixel1=inte_superpixel1.tolist()
	inte_superpixel2=inte_superpixel2.tolist()
	pc_img=cv2.imread(os.path.join(pc_path,img1_n[:-5]+'_pc.png'),-1)
	for super1 in inte_superpixel1:   
		analyze(super1,pc_img,sample_img)
	for super2 in inte_superpixel2:   
		analyze2(super2,pc_img,sample_img)
	cv2.imwrite(os.path.join('./sample_selection/sample_result',img1_n[:-5]+'.png'),sample_img,[int(cv2.IMWRITE_PNG_COMPRESSION),0])  
	print('save the sample selectoin result!')


if __name__=='__main__':
	sample_img=np.copy(pc_map)
	super_segmentation(img1,img2)
	sample_selection(img1,img2)
         
          
    
    
    
