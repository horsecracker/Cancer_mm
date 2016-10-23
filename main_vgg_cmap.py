import tensorflow as tf
import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix

from detector import Detector

from utils_birads import *

train_log='../tensorboard_out/birads_vgg_cam'
test_log='../tensorboard_out/birads_vgg_cam'

logfile='../logfile/log_birads_vgg_cam.txt'
f=open(logfile, 'a+')
f.write('pretrained from vgg \n')


weight_path = 'caffe_layers_value.pickle'
model_path = '../checkpoint_file/birads_vgg_cam'

pretrained_model_path = None #'../models/caltech256/model-0'
n_epochs = 10000
#init_learning_rate = 0.01
init_learning_rate = 0.001
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 64


img_size = 256

#learning_rate= 0.0001
display_step = 20
savemodel_step = 2000
lr_decay_step = 50


dataset_path = '/media/storage3/Study/data/256_ObjectCategories'

caltech_path = '../data/caltech'
trainset_path = '../data/caltech/train.pickle'
testset_path = '../data/caltech/test.pickle'
label_dict_path = '../data/caltech/label_dict.pickle'


###################
'''
if not os.path.exists( trainset_path ):
    if not os.path.exists( caltech_path ):
        os.makedirs( caltech_path )
    image_dir_list = os.listdir( dataset_path )

    label_pairs = map(lambda x: x.split('.'), image_dir_list)
    labels, label_names = zip(*label_pairs)
    labels = map(lambda x: int(x), labels)

    label_dict = pd.Series( labels, index=label_names )
    label_dict -= 1
    n_labels = len( label_dict )

    image_paths_per_label = map(lambda one_dir: map(lambda one_file: os.path.join( dataset_path, one_dir, one_file ), os.listdir( os.path.join( dataset_path, one_dir))), image_dir_list)
    image_paths_train = np.hstack(map(lambda one_class: one_class[:-10], image_paths_per_label))
    image_paths_test = np.hstack(map(lambda one_class: one_class[-10:], image_paths_per_label))

    trainset = pd.DataFrame({'image_path': image_paths_train})
    testset  = pd.DataFrame({'image_path': image_paths_test })

    trainset = trainset[ trainset['image_path'].map( lambda x: x.endswith('.jpg'))]
    trainset['label'] = trainset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
    trainset['label_name'] = trainset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

    testset = testset[ testset['image_path'].map( lambda x: x.endswith('.jpg'))]
    testset['label'] = testset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
    testset['label_name'] = testset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

    label_dict.to_pickle(label_dict_path)
    trainset.to_pickle(trainset_path)
    testset.to_pickle(testset_path)
else:
    trainset = pd.read_pickle( trainset_path )
    testset  = pd.read_pickle( testset_path )
    label_dict = pd.read_pickle( label_dict_path )
    n_labels = len(label_dict)

learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')
'''

####################### new code for data processing #######

loader = DensityLoader()
n_labels =  loader.n_classes 

learning_rate = tf.placeholder( tf.float32, [])
x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
y = tf.placeholder(tf.float32, [None, n_labels])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

#######################
detector = Detector(weight_path, n_labels)

p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( output, labels_tf ), name='loss')

weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
#loss_tf += weight_decay
cost += weight_decay

pred =  tf.nn.softmax(ouput, name='softmax')
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')

optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), grads_and_vars)
#grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )

tf.scalar_summary('cross entropy cost',loss_tf)
tf.scalar_summary('accuracy',accuracy)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(train_log)
test_writer = tf.train.SummaryWriter(test_log)

saver = tf.train.Saver()

confusion_m_all=[]
with tf.InteractiveSession() as sess:

    tf.initialize_all_variables().run()
    if pretrained_model_path:
        print "Pretrained"
        saver.restore(sess, pretrained_model_path)

    #testset = testset.ix[np.random.permutation( len(testset) )]#[:1000]
    #trainset2 = testset[1000:]
    #testset = testset[:1000]

    #trainset = pd.concat( [trainset, trainset2] )
    # We lack the number of training set. Let's use some of the test images

    step = 1
    print('step: {}'.format(step))
    images_test, labels_test = loader.load_test()
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = loader.next_batch(batch_size)
        # Run optimization op (backprop)
        summary,_=sess.run([merged, train_op], feed_dict={learning_rate: init_learning_rate, x: batch_x, y: batch_y})
        train_writer.add_summary(summary,step)

        if step % display_step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            #f=open(logfile, 'r+')
            f.write('\n')
            f.write("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            f.write('\n')
  

        if step % (display_step*2) == 1:
            test_acc_sum = 0.0
            test_steps = len(images_test)/batch_size
            confusion_m_all=[]
            for test_step in range(test_steps):
                temp_acc,yy,summary = sess.run([accuracy,pred,merged], feed_dict={x: images_test[test_step*batch_size:(test_step+1)*batch_size],
                                              y: labels_test[test_step*batch_size:(test_step+1)*batch_size],
                                              keep_prob: 1.})
                test_acc_sum+=temp_acc
                y_true=labels_test[test_step*batch_size:(test_step+1)*batch_size]
                confu_m=confusion_matrix(np.argmax(y_true,1), np.argmax(yy,1)) #,labels=[0,1,2,3])
                #if confu_m.shape==(4,4):
                #print(y_true.shape)
                #print (confu_m)
                confusion_m_all.append(confu_m)
                test_writer.add_summary(summary,step*test_steps+test_step)
                print("Test Accuracy: {} \n".format(test_acc_sum/test_steps))
                confusion_m_average=np.sum(confusion_m_all, axis=0)
                #if step% 5000==1:
                print(confusion_m_average) 
                #f=open(logfile, 'r+')
                f.write("Test Accuracy: {} \n".format(test_acc_sum/test_steps))
                f.write(str(confusion_m_average))
                #f.close

        if step % (savemodel_step) == 1:
            print('at step ' +str(step) + ' model saved. ' )
            save_path=saver.save(sess,save_model_name+'-'+str(step)+'.ckpt')

        if step % (lr_decay_step) == 1:
            init_learning_rate *= 0.99

        step += 1

    f.close()





