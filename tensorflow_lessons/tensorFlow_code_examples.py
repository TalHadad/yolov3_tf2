import tensorflow as tf
#####################
# tensorflow 1 code #
#####################

# 1. set up the variable and operations that define your model
x=tf.placeholder(tf.float,[None,20])
y=tf.placeholder(tf.float,[None,5])
w=tf.placehoder(tf.get_variable('w',shape=(20,5), initializer=tf.initializers.glorot_normal()))
b=tf.get_varable('b', shape=(5,), initializer=tf.initializers.zero())
h=tf.maxmul(x,w)+b

# 2. loss function and optimizer to train the model
loss=tf.losses.mean_squared_error(h,y)
opt=tf.train.GrdientDescengOptimizer(0.001)
train_=opt.minimize(loss)

max_steps=1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(max_steps):

        # 3. run the training operation and calculate the loss
        x_batch,y_batch=next(train_batch)
        _,batch_loss=sess.run([train_op,loss],feed_dict={x:x_batch,y:y_batch})

#####################
# tensorflow 2 code #
#####################

# 1.
import tensorflow as tf
x=tf.Variable([1.,2.],name='x')
print(x) #<tf.Variable 'x:0' shape=(3,) dtype=float32,numpy=array([1.,2.],dtype=float32)>
#Eager execution as default: Use Variables and tensors straight away,
# no need to run an initializer or to launch session object to get their values.

# 2.
tf.keras # as the high-level API

# 3.
# API Cleanup: before there were a few inconsistencies (multiple functions doing very similar things)

#####################################
# tensorflow 2 code in google Colab #
#####################################

# 1.
import google.colab import drive
drive mount('gdrive') # mount your google drive to the notebook (for dataset).

# 2.
myfile=open('gdrive/MyDrive/Colab Notebooks/hello.text') # load a file.
print(myfile.read())

# 3.
!ls # bash commands (to install any package).
!pip install numpy

# 4.
!pip install tensorflow==2 # to upgrade to TensorFlow 2,
# don't forget to restart runtime when complited Runtime-> Restart Runtime).

# 5.
import tensorflow as tf
tf.__version__ # to ensure version 2 installed.

# 6.
# upload jupyter notebooks (File -> Upload notebook, or Colab notebooks) also from GitHub.

# 7.
# download Colab notebooks as jupyter notebook (File -> Download.ipynb)

####################################
# upgrading from TensorFlow 1 to 2 #
####################################

# on TensorFlow.org guide:
# 1. upgrade script - relies heavily on tf.compat.v1 and not transforming your code to use the TensorFlow 2 syntax or idiom.
# 2. upgrading your code to the native TensorFlow 2 style (replace session.run calls, placeholders and so on).

# TensorFlow 1 code:
in_a = tf.placeholder(dtype=tf.float32,shape=(2))
in_b = tf.placeholder(dtype=tf.float32,shape=(2))
def forward(x):
    with tf.variable_scope('matmul',reuse=tf.ATO_REUSE):
        W=tf.get_variable('w',initializer=tf.ones(shape=(2,2)),regulaizer=tf.contrib.layers.l2_regularizer(0.04))
        b=tf.get_variable('b', initializer=tf.zeros(shape=(2)))
        return W*x+b
out_a=forward(in_a)
out_b=forward(in_b)
reg_loss=tf.losses.get_regularization_loss(scope='mantmul')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outs=sess.run([out_a,out_b,reg_loss],feed_dict={in_a:[1,0],in_b:[0,1]})

# TensorFlow 2 code:
W = tf.Variable(tf.ones(shapw=(2,2)),name='W')
b=tf.Variable(tf.zeros(shape=(2,2)),name='b')
@tf.function
def forward(x):
    return W*x+b
out_a=forward([1,0])
print(out_a)
out_b=forward([0,1])
regularizer=tf.keras.regularizers.l2(0.04)
reg_loss=regularizer(W)

# The TensorFlow upgrade function, TF upgrade v2
# (transform to compatible code to TensorFlow 2 but does not take advantage of eager execution)

# 1. create TensorFlow 1 and 2 environments.
$ virtualenv -p python3 TF1
$ virtualenv -p python3 TF2
$ source TF1/bin/activate
(TF1)$ pip install tensorflow==1.14
(TF1)$ deactibate
$ source TF2/bin/activate
(TF2)$ pip install tensorflow==2.0

 # 2. the file exist in the environment
(TF2)$ ls
(TF2)$ vi linear_regression_tf1.py

# 3. the script runs successfully on TensorFlow 1
(TF2)$ source TF1/bin/actibare
(TF1)$ python3 linear_regression_tf1.py

# 4. error when running on TensorFlow 2 (AttributeError)
(TF1)$ source TF2/bin/activate
(TF2)$ python3 linear_regression_tf1.py

#5. adapt the code to TensorFlow 2
(TF2)$ tf_upgrade_v2 --infile linear_regression_tf1.py --outfile linear_regression_tf2.py
(TF2)$ ls
(TF2)$ vi report.txt
(TF2)$ vi linear_regression_tf2.py
(TF2)$ python3 linear_regression_tf2.py # RuntimeError
(TF2)$ vi linear_regression_tf2.py
# add to the code: tf.compact.v1.disable_eager_execution()
# save and close the script :x
(TF2)$ python3 linear_regression_tf2.py # runs successfullt


