import os
import numpy as np
import tensorflow as tf
import scipy.ndimage
from sklearn.utils import shuffle
from PIL import Image
from PIL import ImageEnhance

CLASSES = {'1':0, '2':1, '3':2, '4':3}
# CLASSES = {'1':0, '4':1}

# DataLoader class: need to customize according to your dataset
class DensityLoader(object):
    def __init__(self):
        self.train_data, self.train_labels = load_density_data('./data-density-formatted/train-256/')
        print(self.train_labels[0:])
        self.test_data, self.test_labels = load_density_data('./data-density-formatted/dev-256/')

        self.n_classes = len(CLASSES.keys())
        
        self.num = self.train_data.shape[0]
        self.h = 256
        self.w = 256
        self.c = 1
        
        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.h, self.w, self.c)) 
        labels_batch = np.zeros((batch_size, 4))
        for i in range(batch_size):
            images_batch[i, ...] = distort_image(self.train_data[self._idx]).reshape((self.h, self.w, self.c))
            labels_batch[i, ...] = self.train_labels[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                #self.train_data = shuffle(self.train_data, random_state=20)
                #self.train_labels = shuffle(self.train_labels, random_state=20)
                self.train_data,self.train_labels = shuffle(self.train_data,self.train_labels, random_state=20)

        return images_batch, labels_batch
    
    def load_test(self):
        return self.test_data.reshape((-1, self.h, self.w, self.c)), self.test_labels


def load_density_data(data_path):
    images = []
    labels = []
    n_classes = len(CLASSES.keys())
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if name.split('.')[-1] in {'jpg'}:
                full_path = os.path.join(root, name)
                # im_gray = scipy.misc.imread(full_path, mode='L')
                im_gray = scipy.ndimage.imread(full_path, mode='L')
                #im_gray=(im_gray/255.0)-0.5
 		label_string = full_path.split('/')[-2]
                label = np.zeros(n_classes)
                label[CLASSES[label_string]] = 1.0
                #print im_gray.shape##############
                if im_gray.shape[0]==256 and im_gray.shape[1]==256:
     		  labels.append(label)
                  images.append(im_gray)

    labels,images = shuffle(labels,images)
    #images = shuffle(images)

    X = np.array(images)
    y = np.array(labels)

    print(X.shape)
    print(y.shape)

    return X, y


def distort_image(image):
    
    # flip the image left-right with 50% chances
    if np.random.binomial(1,0.5):
        image=np.fliplr(image)
        
    # rotage the image using PIL rotate funcion
    newimg=Image.fromarray(image)
    max_deg=45
    deg = int(np.random.uniform(0, max_deg))
    newimg=newimg.rotate(deg)
    
    '''
    # change the image contrast/Brightness/sharpness, ehance (1.0) returns orginal iamge
    enchancer1=ImageEnhance.Contrast(newimg)
    c_distort=np.random.normal(loc=0.0, scale=0.2, size=None)
    newimg=enchancer1.enhance(1.0+c_distort)
    
    enchancer2=ImageEnhance.Brightness(newimg)
    b_distort=np.random.normal(loc=0.0, scale=0.2, size=None)
    newimg=enchancer2.enhance(1.0+b_distort)
    '''
    enchancer3=ImageEnhance.Sharpness(newimg)
    s_distort=np.random.normal(loc=0.0, scale=0.2, size=None)
    newimg=enchancer3.enhance(1.0+s_distort)
    
    distorted_img=np.array(newimg)
    
    #distorted_img=tf.image.random_flip_left_right(rotated_image)
    #distorted_img=tf.image.random_brightness(distorted_img)
    
    return distorted_img


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
