## SOFTMAX REGRESSION MODEL

#Importing data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#Implementing regression
x = tf.placeholder("float", [None,784])  #placeholder to hold 2d-tensor representation of mnist images
W = tf.Variable(tf.zeros([784,10]))      #weights: initialized to 0's
b = tf.Variable(tf.zeros([10]))          #biases: initialized to 0's

y = tf.nn.softmax(tf.matmul(x,W)+b)      #Implementation of softmax model!


#Training the model: cost-function is defined using cross-entropy
y_ = tf.placeholder("float", [None,10])        #placeholder to input correct answers
cross_entropy = -tf.reduce_sum(y_*tf.log(y))    #defining loss function i.e cross-entropy

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  #describing how model should be trained
                                                                              #i.e use gradient descent algo to minimize cross_entropy

init = tf.initialize_all_variables()           #defining an operation that initializes all variables created



# Execution: Launching and running the session
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})<== why=0.098?
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# Model Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



