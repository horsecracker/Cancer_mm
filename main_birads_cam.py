'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils_birads import *
import numpy as np
from sklearn.metrics import confusion_matrix

train_log='../tensorboard_out/birads_train_log_cam'
test_log='../tensorboard_out/birads_test_log_cam'
save_model_name='../checkpoint_file/birads_cam'

logfile='../logfile/log_birads_cam.txt'
f=open(logfile, 'a+')
f.write('0.01, 0.1 sttdev initialization of weight, data with whitenning \n')


# Parameters
#learning_rate= 0.0001
learning_rate = 0.00005
training_iters= 20000000
batch_size = 64
display_step = 20
savemodel_step = 2000

# Network Parameters
img_size = 256
n_input = img_size*img_size # data input
n_classes = 2 # total classes (0-3)
#dropout = 0.75 # Dropout, probability to keep units
dropout = 0.75
epsilon = 1e-3

####### logfile for hyperparameter and output check #########
f.write('learning rate %f \n' %learning_rate)
f.write('dropout rate: %f \n' %dropout)
f.write('\n')

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
loader = DensityLoader()

		


def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.01,
           name="conv2d", visualize=False, summarize=False):
    with tf.variable_scope(name):
        #stddev = math.sqrt(6.0)/math.sqrt(float(input_.get_shape().as_list()[-1])+ float(output_dim)
        w = tf.get_variable(name='w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable(name='biases', [output_dim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.bias_add(conv, biases, name='conv')
        return conv



def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',name='pool')

def campool2d(input_, output_dim):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',name='pool')

# Create model
def conv_net(x, weights, biases, keep_prob):

    # Convolution Layer
    
    with tf.variable_scope('conv1') as scope:

        conv1 = conv2d(x,32)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

    with tf.variable_scope('conv2') as scope:
        # Convolution Layer
        conv2 = conv2d(conv1, 64)
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d(conv2, 128)
        # Max Pooling (down-sampling)
        conv3 = maxpool2d(conv3, k=2)

    with tf.variable_scope('conv4') as scope:
        conv4 = conv2d(conv3, 246)
        # Max Pooling (down-sampling)
        conv4 = maxpool2d(conv4, k=2)
    #conv4=tf.nn.dropout(conv4,dropout)

    with tf.variable_scope('conv5') as scope:
        conv5 = conv2d(conv3, 512, k_h=3, k_w=3)
        # Max Pooling (down-sampling)
        conv5 = maxpool2d(conv4, k=2)


    with tf.variable_scope("GAP"):
        gap = tf.reduce_mean( conv5, [1,2], 'input')
        gap =  tf.nn.dropout(gap, keep_prob ,name='dropout')

    with tf.variable_scope('output'):
        gap_w = tf.get_variable(
                    "W",
                    shape=[1024, n_classes],
                    initializer=tf.random_normal_initializer(0., 0.01))
        logits=  tf.matmul( gap, gap_w, name='logits')
        out = tf.nn.softmax(logits,name='softmax')

    return logits,out

# Construct model
print('constructing model')
log,pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(log, y))
# Define loss and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(log, y),name='cross_entropy')
tf.scalar_summary('cross entropy cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.scalar_summary('accuracy',accuracy)

#merge all summaries and define write location
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(train_log)
test_writer = tf.train.SummaryWriter(test_log)


# Initializing the variables
init = tf.initialize_all_variables()
saver=tf.train.Saver()

confusion_m_all=[]
# Launch the graph
print('launch graph')
with tf.Session() as sess:
    sess.run(init)
    step = 1
    print('step: {}'.format(step))
    images_test, labels_test = loader.load_test()
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = loader.next_batch(batch_size)
        # Run optimization op (backprop)
        summary,_=sess.run([merged,optimizer], feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            train_writer.add_summary(summary,step)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
	    #f=open(logfile, 'r+')
	    f.write('\n')
            f.write("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            f.write('\n')
	    #f.close
  

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
		confu_m=confusion_matrix(np.argmax(y_true,1), np.argmax(yy,1),labels=[0,1,2,3])
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

        step += 1

    f.close()
