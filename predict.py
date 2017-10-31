#add your imports here
import ntpath
from sys import argv
from glob import glob
import random
import shelve
import numpy as np
import tensorflow as tf
from cnn import cnn_recognition
from img_seg import *
from scipy import misc
import os

"""
add whatever you think it's essential here
"""
class SymPred():
	def __init__(self,prediction, x1, y1, x2, y2):
		"""
		<x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
		(x1,y1)
			   .--------
			   |	   	|
			   |	   	|
			    --------.
			    		 (x2,y2)
		"""
		self.prediction = prediction
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		
	def __str__(self):
		return self.prediction + '\t' + '\t'.join([
												str(self.x1),
												str(self.y1),
												str(self.x2),
												str(self.y2)])

class ImgPred():
	def __init__(self,image_name,sym_pred_list,latex = 'LATEX_REPR'):
		"""
		sym_pred_list is list of SymPred
		latex is the latex representation of the equation 
		"""
		self.image_name = image_name
		self.latex = latex
		self.sym_pred_list = sym_pred_list
		
	def __str__(self):
		res = self.image_name + '\t' + str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
		for sym_pred in self.sym_pred_list:
			res += str(sym_pred) + '\n'
		return res

def input_wrapper(f,arr=None):
        
        if arr==None:
                image = misc.imread(f)
        else:
                image=arr
        sx,sy = image.shape
        diff = np.abs(sx-sy)

        sx,sy = image.shape
        image = np.pad(image,((sx//8,sx//8),(sy//8,sy//8)),'constant')

        if sx > sy:
                image = np.pad(image,((0,0),(diff//2,diff//2)),'constant')
        else:
                image = np.pad(image,((diff//2,diff//2),(0,0)),'constant')

        #     image = dilation(image,disk(max(sx,sy)/28))
        image = misc.imresize(image,(28,28))
        if np.max(image) > 1:
                image = image/255.

        return image


def get_set(img_data,img_label=None,label_dict=None):
    
        count=0
        for i in range(len(img_data)):
                cur=np.zeros(42)
                if count==0:
                    data=img_data[i].reshape(1,784)
                    if img_label!=None:
                        cur[label_dict[img_label[i]]]=1
                        label=[cur]
                    
                else:
                    data=np.row_stack((data,img_data[i].reshape(1,784)))
                    if img_label!=None:
                        cur[label_dict[img_label[i]]]=1
                        label=np.row_stack((label,cur))
                count+=1
        #         if count>100: break
            
        #     print len(data), len(label)
        if img_label!=None:
                return data, label
        else:
                return data

def norm_result(arr):
    
        maxx=np.max(arr)
        minn=np.min(arr)
        for x in range(len(arr)):
                arr[x]=float(arr[x] - minn)/(maxx- minn)
        arr = arr/sum(arr)
        return arr

def predict(image_path):

        img_data = shelve.open('img_data.db')
        name = img_data['name']
        n2i = img_data['label_dict']
        i2n = img_data['index']

        img_prediction = []

        with tf.Session() as sess:
                    
                clf = cnn_recognition(sess,flag='predict')
                clf.init_network(save=True)
                prior=['pi','=','i','div']
             
                for img_file in image_path:
                    #name_list = img_file.split('_')
                    #if not len(name_list) == 3: continue
                    #print(img_file)
                    
                    x, y, coord, b_box=segment(img_file)
                    merge_group = recog_merge(coord, b_box, x, y)
                    
                    for m in merge_group:
                        img = [input_wrapper(f=None,arr=m[0][0])]
                        data = get_set(img)
                        predict = clf.network(data,save=True)
                        result = i2n[np.argmax(predict)]
                        if result in prior and np.max(predict)>3:
                            key_set=m[1]
                            
                            for k in range(1,len(key_set)):
                                if coord[key_set[0]] is None: continue
                                print(len(coord[key_set[0]]),'1',type(coord[key_set[0]]))
                                print(len(coord[key_set[k]]),'2')
                #                     coord[key_set[0]]=coord[key_set[0]]+coord[key_set[k]]
                                coord[key_set[0]]= np.row_stack((coord[key_set[0]],coord[key_set[k]]))
                                coord[key_set[k]] = None
                                print(len(coord[key_set[0]]),'3')
                                
                    img_group, img_coord = output_img(coord, x, y)
                    
                    content = []
                    count=0
                    for i in range(len(img_group)):
                        img=[input_wrapper(f=None,arr=img_group[i])]
                        data=get_set(img)
                        predict=clf.network(data)
                        result=i2n[np.argmax(predict)]
                        ans=np.max(predict)
                        if ans < 3: continue
                        count += 1
                        print(result,ans,img_coord[i],'result')
                        
                        predict_img = SymPred(result, img_coord[i][0], img_coord[i][2], img_coord[i][1], img_coord[i][3])
                        #print(predict_img.__str__())
                        content.append(predict_img.__str__())

                    img_prediction.append(content)
                    
        return img_prediction

if __name__ == '__main__':
        
	image_folder_path = argv[1]
	if len(argv) == 3:
		isWindows_flag = True
	if isWindows_flag:
		image_paths = glob(image_folder_path + '/*png')
	else:
		image_paths = glob(image_folder_path + '\\*png')

	results = predict(image_paths)
	with open('predictions.txt','a') as fout:
		for i in range(len(results)):
			name = ntpath.basename(image_paths[i])
			ans = ImgPred(name, results[i])
			#print(ans.__str__())
			fout.write(ans.__str__())
