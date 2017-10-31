import os
import random
from scipy import misc
import scipy.ndimage as ndi
import numpy as np
from skimage.morphology import dilation,disk
import shelve

def input_wrapper(f):
    image = misc.imread(f)
    sx,sy = image.shape
    diff = np.abs(sx-sy)

    sx,sy = image.shape
    image = np.pad(image,((sx//8,sx//8),(sy//8,sy//8)),'constant')
    if sx > sy:
        image = np.pad(image,((0,0),(diff//2,diff//2)),'constant')
    else:
        image = np.pad(image,((diff//2,diff//2),(0,0)),'constant')
    
    image = dilation(image,disk(max(sx,sy)/28))
    image = misc.imresize(image,(28,28))
    
    if np.max(image) > 1:
        image = image/255.0

    return image

#add noise image
def add_noise(img):
    
    SNR = 0.995
    noiseNum = int((1 - SNR)*img.shape[0]*img.shape[1])

    for i in range(noiseNum):
        randX = np.random.random_integers(0,img.shape[0]-1)  
        randY = np.random.random_integers(0,img.shape[1]-1)  

        if np.random.random_integers(0,1)==0:  
            img[randX,randY]=0  
        else:  
            img[randX,randY]=255

    img = img/255.0

    #imshow(img)
    #show()
    return img
        
def get_data(path):
    
    imgs = os.listdir(path)
    data=[]
    label=[]
    name=[]
    l2i=dict()
    i2l=[]
    count=0

    for img in imgs:
        #print(img)
        name_list = img.split('_')
        if len(name_list) < 4: continue
        
        print(name_list[1])
       
        data.append(input_wrapper(path+'/'+img))
        label.append(name_list[3])
        
        if name_list[3] not in l2i:
            l2i[name_list[3]] = count
            i2l.append(name_list[3])
            count+=1
            
        name.append(img)

    for i in range(15):
        rand = np.random.random_integers(0,len(data)-1)
        n = add_noise(np.zeros((28,28)))
        data.insert(rand, n)
        label.insert(rand, "noise")
        name.insert(rand, "noise_"+str(i))
        
    i2l.append("noise")
    l2i["noise"] = count

    img_data=shelve.open('img_data.db')
    img_data['data']=data
    img_data['label']=label
    img_data['label_dict']=l2i
    img_data['index']=i2l
    img_data['name']=name
    
    print(len(img_data['data']), len(img_data['label']))
    print(img_data['label_dict'])
    img_data.close()

get_data('annotated')
