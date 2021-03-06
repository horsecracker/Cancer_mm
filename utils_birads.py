import os
import numpy as np
import tensorflow as tf
import scipy.ndimage
from sklearn.utils import shuffle
from ImageAugmenter import ImageAugmenter

CLASSES = {'0':0, '1':0, '2':0, '3':0, '4':1, '5':1, '6':1, '9':0}
# CLASSES = {'1':0, '4':1}

# DataLoader class: need to customize according to your dataset
class DensityLoader(object):
    def __init__(self, logfile, data_3d = True):
        self.train_data, self.train_labels = load_density_data('../birads_dataset/train-sq-512/', need_3d = data_3d )
        print(self.train_labels[0:])
        self.test_data, self.test_labels = load_density_data('../birads_dataset/dev-sq-512/', need_3d = data_3d )

        #self.n_classes = len(set(CLASSES.values()))

        self.h = 512
        self.w = 512
        if data_3d:
            self.c = 1 
        else:
            self.c = 1
        
        self._idx = 1

        self.classes_map = CLASSES
        self.n_classes = len(set(self.classes_map.values()))
        #self.load_small_data_for_debug = FLAGS.load_small_data_for_debug
        self.should_enforce_class_balance = True #FLAGS.should_enforce_class_balance
        #self.verbose = FLAGS.verbose
        #self.path_to_image_directory = FLAGS.path_to_image_directory
        #self.MODEL_CLASS = MODEL_CLASS
        #self.image_width, self.image_height, self.c = MODEL_CLASS.get_image_dimensions()

        # Load Data
        #self.load_data_from_metadata()

        self.print_all_label_statistics(logfile)
        if self.should_enforce_class_balance:
            print("Enforcing Class balance")
            self.enforce_class_balance()
            self.print_all_label_statistics(logfile)

        self.trainnum = self.train_data.shape[0]
        self.testnum = self.test_data.shape[0]
        
        #self.training_examples_count = self.train_labels.shape[0]
        #self.dev_examples_count = self.dev_labels.shape[0]
        #self.test_examples_count = self.test_labels.shape[0]

        #self.n_train_examples, self.n_dev_examples, self.n_test_examples = self.train_data[0].shape[0], self.dev_data[0].shape[0], self.test_data[0].shape[0]
        self.augment_training_data = False
        self.augmenter = ImageAugmenter(self.w, self.h, # width and height of the image (must be the same for all images in the batch)
                           hflip=True,    # flip horizontally with 50% probability
                           vflip=True,
                           scale_to_percent=1.2, # scale the image to 70%-130% of its original size
                           scale_axis_equally=False, # allow the axis to be scaled unequally (e.g. x more than y)
                           rotation_deg=10,    # rotate between -25 and +25 degrees
                           shear_deg=5,       # shear between -10 and +10 degrees
                           translation_x_px=20, # translate between -5 and +5 px on the x-axis
                           translation_y_px=20  # translate between -5 and +5 px on the y-axis
                           )            
    
    def print_label_statistics(self, labels, logfile, labels_label):
        f=open(logfile, 'a+')

        class_count = {key: 0 for key in set(self.classes_map.values())}
        for label in labels:
            class_count[np.argmax(label)] += 1
        print("Class Balance for {}: {}. Total #: {}".format(labels_label, class_count, len(labels)))
        f.write("Class Balance for {}: {}. Total #: {}\n ".format(labels_label, class_count, len(labels)))
        f.close()
        return class_count


    def print_all_label_statistics(self, logfile):
        self.print_label_statistics(self.train_labels,logfile, "Train")
        #self.print_label_statistics(self.dev_labels, "Dev")
        self.print_label_statistics(self.test_labels, logfile, "Test")

    def enforce_class_balance(self):
        #self.train_data, self.train_labels = self.enforce_class_balance_helper(self.train_data, self.train_labels)
        #self.dev_data, self.dev_labels = self.enforce_class_balance_helper(self.dev_data, self.dev_labels)
        self.test_data, self.test_labels = self.enforce_class_balance_helper(self.test_data, self.test_labels)

    def enforce_class_balance_helper(self, data, labels):
        class_count = {key: 0 for key in set(self.classes_map.values())}
        for i in range(labels.shape[0]):
            label = labels[i][...]
            class_count[np.argmax(label)] += 1
        min_class_count = min(class_count.values())

        image_data = data
        #image_data, additional_data = data

        image_data_new = []
        #additional_data_new = []
        labels_new = []
        for cl, count in class_count.iteritems():
            label_target = [1 if i == cl else 0 for i in range(len(set(class_count.values())))]
            indicies = np.where(labels == label_target)[0]
            indicies = list(set(indicies))
            cur_count = 0
            for index in indicies:
                if cur_count < min_class_count:
                    image_data_new.append(image_data[index][...])
                    #additional_data_new.append(additional_data[index][...])
                    labels_new.append(labels[index][...])
                    cur_count += 1

        image_data_new = np.array(image_data_new)
        #additional_data_new = np.array(additional_data_new)
        #data_new = (image_data_new, additional_data_new)
        labels_new = np.array(labels_new)
        return image_data_new, labels_new

    def augment_images(self, images):
        augmented_images = ((images+0.5)*255.0).astype('uint8')
        augmented_images = self.augmenter.augment_batch(augmented_images) - 0.5
        return augmented_images


    def next_batch(self, batch_size, data_group='train'):
        images_batch = np.zeros((batch_size, self.h, self.w, self.c)) 
        labels_batch = np.zeros((batch_size, self.n_classes))
        for i in range(batch_size):
            images_batch[i, ...] = self.train_data[self._idx].reshape((self.h, self.w, self.c))
            labels_batch[i, ...] = self.train_labels[self._idx]
            
            self._idx += 1
            if self._idx == self.trainnum:
                self._idx = 0
                #self.train_data = shuffle(self.train_data, random_state=20)
                #self.train_labels = shuffle(self.train_labels, random_state=20)
                self.train_data,self.train_labels = shuffle(self.train_data,self.train_labels, random_state=20)

        if data_group == 'train' and self.augment_training_data:
            images_batch = self.augment_images(images_batch)

        return images_batch, labels_batch
    
    def load_test(self):
        #print('test image size of {} :'.format(str(test_data.shape)))
        return self.test_data.reshape((-1, self.h, self.w, self.c)), self.test_labels



def load_density_data(data_path, need_3d = True):
    images = []
    labels = []
    n_classes = len(set(CLASSES.values()))
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if name.split('.')[-1] in {'jpg', 'png'}:
                full_path = os.path.join(root, name)
                # im_gray = scipy.misc.imread(full_path, mode='L')
                im_gray = scipy.ndimage.imread(full_path, mode='L')
                im_gray =(im_gray/255.0)-0.5
                label_string = full_path.split('/')[-2]
                label = np.zeros(n_classes)
                label[CLASSES[label_string]] = 1.0
                #print im_gray.shape##############
                #if im_gray.shape[0]==256 and im_gray.shape[1]==256:
                labels.append(label)
                if need_3d:
                    im_gray=np.reshape(im_gray,(256,256,1))
                    im_gray=np.concatenate([im_gray,im_gray,im_gray],axis=2)    
                    #im_gray=np.concatenate((img_gray,img_gray,img_gray),axis=2)
                images.append(im_gray)

    labels,images = shuffle(labels,images)
    #images = shuffle(images)

    X = np.array(images)
    y = np.array(labels)

    print(X.shape)
    print(y.shape)

    return X, y



