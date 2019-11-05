import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # 设置随机种子
# 定义数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
nb_classes = 10
# 定义占位符
X = tf.placeholder("float", shape=[None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, nb_classes], name='Y')
# 权重和偏置
W = tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
# 预测模型
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
logits = tf.add(tf.matmul(X, W), b, name='logits')

hypothesis = logits  # tf.nn.softmax(logits)
# 代价或损失函数
# cost = tf.reduce_mean(tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# 梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = train.minimize(cost, name='train')
# 准确率计算
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 创建会话
sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())  # 全局变量初始化
# 迭代训练
training_epochs = 10
batch_size = 100
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch
    # 显示损失值收敛情况
    print(epoch, avg_cost)
saver.save(sess, './checkpoint_dir/MyModel')

# 准确率
print("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images[:400], Y: mnist.test.labels[:400]}))
# 在测试集中随机抽一个样本进行测试
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
plt.imshow(mnist.test.images[r].reshape(28, 28), cmap='Greys')
plt.show()
while True:
    str = input()
    try:
        if str == 'q':
            break
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
        plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys')
        plt.show()
    except:
        continue
sess.close()
