import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data")
# 参数
batch_size = 100 #mini-batch批大小
n_steps = 28 #时间步数（序列长度）
n_inputs = 28 #输入数据长度
n_neurons = 100 #隐藏状态，神经元个数
n_outputs = 10 #输出10分类
learning_rate = 0.001 #学习率
n_epochs = 3 #训练大周期
# 输入输出占位符
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) #三维数据(?, 28, 28)
y = tf.placeholder(tf.int32, [None]) # 一维输出，标签0-9
# 模型使用最简单的BasicRNNcell
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)  #outputs(?, 28, 100) states(?, 100)
# logits = fully_connected(states, n_outputs, activation_fn=None)
logits = fully_connected(outputs[:,-1], n_outputs, activation_fn=None) #用最后一个时间步的输出
# 代价或损失函数
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# 计算准确率，只有等于y才是对的，其它都错
correct = tf.nn.in_top_k(logits, y, 1)  #每个样本的预测结果的前k个最大的数里面是否包含targets预测中的标签
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):  # 55000
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs)) #(?, 28, 28)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images.reshape((-1, n_steps, n_inputs)),
                                            y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
