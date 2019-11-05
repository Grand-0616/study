import tensorflow as tf

c1 = tf.constant(9.5,dtype=tf.float32)#常量
c2 = tf.constant(10,dtype=tf.int32)#常量
a = tf.Variable(5.8)#变量
b = tf.Variable(2.9)
sum = tf.Variable(0, name="sum")
result = tf.Variable(1, name="result")

f = tf.Variable([[2., 5., 4.],
                 [1., 3., 6.]])
f2 = tf.Variable([[2., 3.],
                 [1., 2.],
                 [3., 1.]])
vector1 = tf.constant([3.,3.]) #这里只有一对中括号 []，就是向量
vector2 = tf.constant([1.,2.])
result3 = tf.multiply(vector1,vector2) #向量乘法。tf.multiply()
result4 = tf.multiply(vector2,vector1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
r3 = sess.run(result3)
r4 = sess.run(result4)
print(r3,'\n',r4)

#tf.add 加法函数
print("加法", sess.run(tf.add(a,b))) # 求和
#tf.subtract 减法函数
print("减法", sess.run(tf.subtract(a,c1))) #减法
#tf.multiply 乘法函数
print("乘法",sess.run(tf.multiply(a,b))) # 乘积
#tf.divde   除法函数
print("除法",sess.run(tf.divide(a,2.0))) #除法
print("除法",sess.run(tf.div(a,2.0))) #除法

#tf.assign 赋值函数
for i in range(101):
    sess.run(tf.assign(sum, tf.add(sum, i)))
print('1到100的和：', sess.run(sum))

for i in range(1, 11):
    sess.run(tf.assign(result, tf.multiply(result, i)))
print('1到10的乘积：', sess.run(result))

print("f:" ,sess.run(f))
print("转置：", sess.run(tf.transpose(f))) #矩阵转置
print("叉乘：", sess.run(tf.multiply(f, 2))) #叉乘
print("矩阵乘：", sess.run(tf.matmul(f,f2))) #矩阵乘法
print("求和reduce_sum所有", sess.run(tf.reduce_sum(f)))
print("求和reduce_sum按行", sess.run(tf.reduce_sum(f, axis=1)))
print("求和reduce_sum按列", sess.run(tf.reduce_sum(f, axis=0)))

print('行向量最大值的下标argmax=', sess.run(tf.argmax(f, axis=1)))             #行向量最大值的下标
print('类型转换（布尔到浮点数）cast  ',sess.run(tf.cast(1 > 0.5, dtype=tf.float32)))  #类型转换
print('cast  ',sess.run(tf.cast(0.1 > 0.5, dtype=tf.float32)))  #类型转换
print('比较（真） ', sess.run(tf.equal(1.0,1))) #比较
print('比较（假） ', sess.run(tf.equal(1.0,1.0))) #比较
print('比较（真） ', sess.run(tf.equal(0, False))) #比较
print('所有元素平均值reduce_mean=', sess.run(tf.reduce_mean(f)))      #所有元素平均值
print('列向量平均值reduce_mean(0)=', sess.run(tf.reduce_mean(f,0))) #列向量平均值
print('行向量平均值reduce_mean(1)=', sess.run(tf.reduce_mean(f,1))) #行向量平均值
