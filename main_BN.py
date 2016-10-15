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
from utils import *
import numpy as np
from sklearn.metrics import confusion_matrix

train_log='../tensorboard_out/density_train_log-BN'
test_log='../tensorboard_out/density_test_log-BN'
save_model_name='../density_BN.ckpt'

logfile='../logfile/log_BN.txt'
f=open(logfile, 'a+')
f.write('Model using Batch Normalization, data with whitenning \n')

loader = DensityLoader()

# Parameters
#learning_rate= 0.0001
learning_rate = 0.0001
training_iters= 20000000
batch_size = 64
display_step = 20

# Network Parameters
img_size = 256
n_input = img_size*img_size # data input
n_classes = 4 # total classes (0-3)
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

		
# Create some wrappers for simplicity

def conv2d_BN(x, W, scale, beta, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    #x = tf.nn.bias_add(x, b)
    batch_mean, batch_var = tf.nn.moments(x,[0,1,2])
    #scale2 = tf.Variable(tf.ones([100]))
    #beta2 = tf.Variable(tf.zeros([100]))
    BN = tf.nn.batch_normalization(x,batch_mean,batch_var,beta,scale,epsilon)
    return tf.nn.sigmoid(BN)
    
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, scale, beta, biases, dropout):

    # Convolution Layer
    conv1 = conv2d_BN(x, weights['wc1'], scale['sc1'], beta['bt1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    
    # Convolution Layer
    conv2 = conv2d_BN(conv1, weights['wc2'],scale['sc2'], beta['bt2'])
    # Max Pooling (down-sampling)
    
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d_BN(conv2, weights['wc3'], scale['sc3'], beta['bt3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)


    conv4 = conv2d_BN(conv3, weights['wc4'], scale['sc4'], beta['bt4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)
    #conv4=tf.nn.dropout(conv4,dropout)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    #fc1=tf.nn.sigmoid(fc1)

    # Output, class prediction
    logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out = tf.nn.softmax(logits)
    return logits,out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 5x5 conv, 64 inputs, 128 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # 5x5 conv, 128 inputs, 256 outputs
    'wc4': tf.Variable(tf.random_normal([5, 5, 128, 256])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(img_size/16)*(img_size/16)*256, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

scale = {
    'sc1': tf.Variable(tf.ones([32])),
    'sc2': tf.Variable(tf.ones([64])),
    'sc3': tf.Variable(tf.ones([128])),
    'sc4': tf.Variable(tf.ones([256])),
    #'sd1': tf.Variable(tf.zeros([1024])),
    #'out': tf.Variable(tf.zeros([n_classes]))
}

beta = {
    'bt1': tf.Variable(tf.zeros([32])),
    'bt2': tf.Variable(tf.zeros([64])),
    'bt3': tf.Variable(tf.zeros([128])),
    'bt4': tf.Variable(tf.zeros([256])),
    #'bd1': tf.Variable(tf.zeros([1024])),
    #'out': tf.Variable(tf.zeros([n_classes]))
}

biases = { 
    'bd1': tf.Variable(tf.zeros([1024])),
    'out': tf.Variable(tf.zeros([n_classes]))
    }

# Construct model
print('constructing model')
log,pred = conv_net(x, weights, scale, beta, biases, dropout)

# Define loss and optimizer
#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(log, y))
# Define loss and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(log, y))
tf.scalar_summary('cross entropy cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.scalar_summary('accuracy',accuracy)
confusion_m_all=[]

#merge all summaries and define write location
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(train_log)
test_writer = tf.train.SummaryWriter(test_log)


# Initializing the variables
init = tf.initialize_all_variables()

saver=tf.train.Saver()
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
                test_writer.add_summary(summary,step)
	    print("Test Accuracy: {} \n".format(test_acc_sum/test_steps))
            confusion_m_average=np.sum(confusion_m_all, axis=0)
	    #if step% 5000==1:
 	    print(confusion_m_average) 
            #f=open(logfile, 'r+')
 	    f.write("Test Accuracy: {} \n".format(test_acc_sum/test_steps))
 	    f.write(str(confusion_m_average))
 	    #f.close

        step += 1
    save_path=saver.save(sess,save_model_name)
    f.close()
