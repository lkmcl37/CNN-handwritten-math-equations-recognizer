import tensorflow as tf 
import numpy as np
import shelve

class cnn_recognition():
    def __init__(self,sess,cat_num=42,model_path='./model.ckpt',flag=True):
        self.sess = sess
        self.flag = flag
        self.cat_num =cat_num
        self.model_path = model_path
            
    def weight_variable(self, shape,n):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial,name=n)
        initial = tf.truncated_normal(shape, stddev=0.01)
        var = tf.Variable(initial)
        weight_decay = tf.multiply(tf.nn.l2_loss(var), 1e-5, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        
        return var
    
    def bias_variable(self,shape,n):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name=n)
      
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    def batch_norm_layer(self, inputs, decay = 0.9):
        
        epsilon = 1e-5
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        
        train_mean = tf.assign(pop_mean,
                pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                pop_var * decay + batch_var * (1 - decay))
        
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon),batch_mean,batch_var
  
    def init_network(self,save=False):
        
        self.x = tf.placeholder(tf.float32, [None, 784])                      
        self.y_actual = tf.placeholder(tf.float32, shape=[None, self.cat_num])   
        
        x_image = tf.reshape(self.x, [-1,28,28,1])         
        W_conv1 = self.weight_variable([5, 5, 1, 32],'W_conv1')      
        b_conv1 = self.bias_variable([32],'b_conv1')
        
        tmp_1,_,_ = self.batch_norm_layer(self.conv2d(x_image, W_conv1)+b_conv1)
        h_conv1 = tf.nn.relu(tmp_1)  
        h_pool1 = self.max_pool(h_conv1)                                
        
        W_conv2 = self.weight_variable([5, 5, 32, 64],'W_conv2')
        b_conv2 = self.bias_variable([64],'b_conv2')
        
        tmp_2,_,_ = self.batch_norm_layer(self.conv2d(h_pool1, W_conv2)+b_conv2)
        h_conv2 = tf.nn.relu(tmp_2)     
        h_pool2 = self.max_pool(h_conv2)                                  
        
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024],'W_fc1')
        b_fc1 = self.bias_variable([1024],'b_fc1')
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])              
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    
        
        self.keep_prob = tf.placeholder("float") 
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)                  
        
        W_fc2 = self.weight_variable([1024, self.cat_num],'W_fc2')
        b_fc2 = self.bias_variable([self.cat_num],'b_fc2')
        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        if save:
            saver = tf.train.Saver()
            saver.restore(self.sess,self.model_path)
            
    def network(self, data, label=None, save=False):
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_actual, logits=self.y_conv))
        tf.add_to_collection('losses', cross_entropy)   
        cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_actual,1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#         with self.sess as sess:
        sess = self.sess
            
        if self.flag=='train':
            sess.run(tf.global_variables_initializer())
            batch=0
            bsize=50
            for i in range(200):          
                train_acc = accuracy.eval(feed_dict={self.x:data[batch:batch+bsize], self.y_actual:label[batch:batch+bsize], self.keep_prob: 1.0})
                print('step',i,'training accuracy',train_acc)
                
                train_step.run(feed_dict={self.x: data[batch:batch+bsize], self.y_actual: label[batch:batch+bsize], self.keep_prob: 1.0})
                batch=(batch+50)%len(data)
                
            saver = tf.train.Saver()
            saver_path = saver.save(sess, self.model_path)
            print("Model saved in file: ", saver_path)

            
        elif self.flag=='predict':
            predict = sess.run(self.y_conv, feed_dict = {self.x:data[0:], self.keep_prob:1})
            train_size=0
#                 y_conv,result = sess.run([self.y_conv,tf.argmax(self.y_conv,1)],feed_dict={self.x:data[train_size:], self.keep_prob: 1.0})
            return predict
#                 return y_conv,result
        
        else:
            saver = tf.train.Saver()
            saver.restore(sess,self.model_path)
            test_acc,result,cor = sess.run([accuracy,tf.argmax(self.y_conv,1),tf.argmax(self.y_actual,1)],feed_dict={self.x:data, self.y_actual:label, self.keep_prob: 1.0})
            print('Final Accuracy',test_acc)
            return test_acc,result,cor

def get_set(img_data, img_label,label_dict):
    
    count=0
    for i in range(len(img_data)):
        cur = np.zeros(42)
        if count==0:
            data=img_data[i].reshape(1,784)
            cur[label_dict[img_label[i]]]=1
            print(img_label[i])
            label=cur
            
        else:
            data=np.row_stack((data,img_data[i].reshape(1,784)))
            cur[label_dict[img_label[i]]]=1
            label=np.row_stack((label,cur))
        count+=1
        
#         if count>100: break
        print(count),'get image'
            
    print(len(data), len(label))
    return data, label

# img_data=shelve.open('img_data.db')
# data_set,label_set=get_set(img_data['data'], img_data['label'], img_data['label_dict'])
# 
# sess=tf.Session()

#print("training...")
#clf=cnn_recognition(sess,flag='train')
#clf.init_network()
#clf.network(data_set[0:3000], label_set[0:3000], train_size=3000)

# clf=cnn_recognition(sess,flag='test')
# clf.init_network()
# clf.network(data_set, label_set, train_size=3000)
