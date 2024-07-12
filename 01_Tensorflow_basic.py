"""
- tensorflow ver1.x 작업 환경
- Graph 모델 정의 & 실행  
"""

# Tensorflow code 
import tensorflow.compat.v1 as tf # ver2.x 환경에서 ver1.x 사용
tf.disable_v2_behavior() # ver2.x 사용 안함 

''' Graph 모델 정의 ''' 

# 상수 정의 
x = tf.constant(10)  
y = tf.constant(20)  

# 식 정의 
z = tf.add(x, y) # z = x + y
'''
<Tensorflow 사칙연산 함수>  
add = tf.Variable(a + b)
subtract = tf.Variable(a - b)
multiply = tf.Variable(a * b)
divide = tf.Variable(a / b)
'''
print('z=', z)
# z= Tensor("Add:0", shape=(), dtype=int32)

''' Graph 모델 실행 ''' 
with tf.Session() as sess : # 세션 생성
    # 처리기 할당  
    print('x=', sess.run(x)) 
    print('y=', sess.run(y))
    print('z=', sess.run(z)) # z= 30

###########################################################################################################

"""
Tensorflow 상수와 변수 정의 
"""

# Tensorflow code 
import tensorflow.compat.v1 as tf # ver1.x -> ver2.x 마이그레이션 
tf.disable_v2_behavior() # ver2.x 사용 안함 

''' Graph 모델 정의 '''
# 상수 정의  
x = tf.constant([1.5, 2.5, 3.5]) # 수정 불가 
print(x) # Tensor("Const_3:0", shape=(3,), dtype=float32)

# 변수 정의  
y = tf.Variable([1.0, 2.0, 3.0]) # 수정 가능   
print(y) # <tf.Variable 'Variable_1:0' shape=(3,) dtype=float32_ref>

''' Graph 모델 실행 '''
with tf.Session() as sess : # 세션 객체 생성     
    print('x =', sess.run(x)) # 상수 실행 
    
    sess.run(tf.global_variables_initializer()) # 변수 초기화 
    print('y=', sess.run(y)) # 변수 실행
    '''
    x = [1.5 2.5 3.5]
    y= [1. 2. 3.]
    '''

###########################################################################################################


"""
Tensorboard 
 - Graph 모델 시각화 도구
 - Tensor의 연산 흐름 확인  
"""

import tensorflow.compat.v1 as tf # ver1.x 사용  
tf.disable_v2_behavior() # ver2.x 사용 안함 

# tensorboard 초기화 
tf.reset_default_graph()


''' Graph 모델 정의 '''
 
# 상수 정의 
x = tf.constant(1)
y = tf.constant(2)

# 사칙연산식 정의 
a = tf.add(x, y, name='a') # a = x + y
b = tf.multiply(a, 6, name='b') # b = a * 6
c = tf.subtract(20, 10, name='c') # c = 20 - 10
d = tf.div(c, 2, name = 'd') # d = c / 2

g = tf.add(b, d, name='g') # g = b + d
h = tf.multiply(g, d, name='h') # h = g * d

''' Graph 모델 실행 '''

with tf.Session() as sess :
    h_calc = sess.run(h) # device 할당 : 연산 
    print('h = ', h_calc) # h =  115
    
    # tensorboard graph 생성
    tf.summary.merge_all() # Graph 모으기  
    writer = tf.summary.FileWriter("C:/ITWILL/7_Tensorflow/graph", sess.graph)
    writer.close()


###########################################################################################################


"""
name_scope 이용
 - 영역별 tensorboard 시각화 
"""

import tensorflow.compat.v1 as tf # ver1.x 사용 
tf.disable_v2_behavior() # ver2.x 사용 안함 

# tensorboard 초기화 
tf.reset_default_graph()

''' Graph 모델 정의 '''

# name : 한글, 공백, 특수문자 사용불가 
X = tf.constant(5.0, name = 'x_data') # 입력변수 
a = tf.constant(10.1, name = 'a') # 기울기 
b = tf.constant(4.45, name = 'b') # 절편 
Y = tf.constant(55.0, name = 'y_data') # 출력(정답)변수 

# name_scope : 한글, 공백, 특수문자 사용불가  
with tf.name_scope('regress_model') as scope :
    model = (X * a) + b # 회귀방정식 : 예측치  
    
with tf.name_scope('model_error') as scope :
    model_err = model - Y # err = 측치-정답 

with tf.name_scope('model_eval') as scope :
    square = tf.square(model_err) # 오차 제곱 
    mse = tf.reduce_mean(square) # 오차 제곱 평균 

''' Graph 모델 실행  '''
with tf.Session() as sess :
    # 각 영역별 실행 결과 
    print('Y = ', sess.run(Y))
    y_pred = sess.run(model)
    print('y pred =', y_pred)
    err = sess.run(model_err)
    print('model error =', err)
    print('MSE = ', sess.run(mse))    
    
    # tensorboard graph 생성 과정 
    tf.summary.merge_all() # 상수,식 모으는 역할 
    writer = tf.summary.FileWriter("C:/ITWILL/7_Tensorflow/graph", sess.graph)
    writer.close()


###########################################################################################################


"""
1. Tensorflow 상수와 변수 
2. 즉시 실행(eager execution) 모드
 - session 사용 없이 자동으로 컴파일 
 - python 처럼 즉시 실행하는 모드 제공(python 코드 사용 권장)
 - API 정리 : tf.global_variables_initializer() 삭제됨 
"""


import tensorflow as tf # ver 2.x
print(tf.__version__) # 2.10.0


# 즉시 실행 모드 
tf.executing_eagerly() # 기본(default)으로 활성화 됨  


# 상수 정의 & 실행  
x = tf.constant(value = [1.5, 2.5, 3.5]) # 1차원   
print('x =', x) 
# x = tf.Tensor([1.5 2.5 3.5], shape=(3,), dtype=float32)

# 변수 정의 & 실행  
y = tf.Variable([1.0, 2.0, 3.0]) # 1차원  
print('y =', y)
# y = <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>
print(y.numpy()) # [1. 2. 3.]


# 식 정의 & 실행 : 상수 or 변수 참조 
mul = tf.math.multiply(x, y) # x * y 
print('mul =', mul) 
# tf.Tensor([ 1.5  5.  10.5], shape=(3,), dtype=float32)
val = mul.numpy()
print(val) # [ 1.5  5.  10.5]

###########################################################################################################

"""
@tf.function 함수 장식자
 - ver2.x에서 그래프 모드 지원 
 - 함수내에서 python code 작성 지원
 - 적용 시 : 모델의 연속 속도가 빨라짐(성능 최적화) 
"""

import tensorflow as tf

def add_eager_mode(a, b): # 즉시 실행 모드 
    return a + b


@tf.function # 그래프 모드 지원 
def add_graph_mode(a, b):
    return a + b


# 실인수 
a = tf.constant(2)
b = tf.constant(3)


# 즉시 실행 모드
result_eager = add_eager_mode(a, b) # (2, 3)
print("Eager mode result:", result_eager)  
# Eager mode result: tf.Tensor(5, shape=(), dtype=int32)

# 그래프 모드
result_graph = add_graph_mode(a, b)
print("Graph mode result:", result_graph)  
# Graph mode result: tf.Tensor(5, shape=(), dtype=int32)

###########################################################################################################

'''
Tensor 정보 제공 함수 
 1. tensor shape
 2. tensor rank
 3. tensor size
 4. tensor reshape 
'''

import tensorflow as tf
print(tf.__version__) # 2.10.0

scala = tf.constant(1234) # 상수 
vector = tf.constant([1,2,3,4,5]) # 1차원 
matrix = tf.constant([ [1,2,3], [4,5,6] ]) # 2차원
cube = tf.constant([[ [1,2,3], [4,5,6], [7,8,9] ]]) # 3차원 

dir(scala)

print(scala)
print(vector)
print(matrix)
print(cube)

# 1. tensor shape 
print('\ntensor shape')
print(scala.get_shape()) # () scalar.shape
print(vector.get_shape()) # (5,)
print(matrix.get_shape()) # (2, 3)
print(cube.get_shape()) # (1, 3, 3)

  
# 2. tensor rank
print('\ntensor rank')
print(tf.rank(scala)) # 0
print(tf.rank(vector)) # 1
print(tf.rank(matrix)) # 2
print(tf.rank(cube)) # 3

# 3. tensor size
print('\ntensor size')
print(tf.size(scala)) # 1
print(tf.size(vector)) # 5
print(tf.size(matrix)) # 6
print(tf.size(cube)) # 9

dir(tf)

# 4. tensor reshape
cube.shape # TensorShape([1, 3, 3])

cube_2d = tf.reshape(cube, (3,3))
print(cube_2d.shape) # (3, 3)

###########################################################################################################

'''
 <수학 관련 주요 함수> 
version 1.x   -> version 2.x
tf.add() -> tf.math.add() 변경 
tf.subtract() -> tf.math.subtract() 변경 
tf.multiply() -> tf.math.multiply() 변경 
tf.div() -> tf.math.divide() 변경 
tf.mod() : 나머지 -> tf.math.mod() 변경 
tf.abs() : 절대값 -> tf.math.abs() 변경 
tf.square() : 제곱  -> tf.math.square() 변경
tf.sqrt() : 제곱근  -> tf.math.sqrt() 변경
tf.round() : 반올림  -> tf.math.round() 변경
tf.pow() : 거듭제곱 -> tf.math.pow() 변경
tf.exp() : 지수값 -> tf.math.exp() 변경
tf.log() : 로그값 -> tf.math.log() 변경
'''

import tensorflow as tf # tf.math.xxxx()

# 상수 정의 & 실행 
x = tf.constant([1,2,-3,4])
y = tf.constant([5,6,7,8])


# 덧셈/뺄샘/나눗셈/곱셈
print(tf.math.add(x, y, name='adder')) # [ 6  8  4 12]
print(tf.math.subtract(x, y, name='adder')) # [ -4  -4 -10  -4]
print(tf.math.multiply(x, y, name='adder'))
print(tf.math.divide(x, y, name='divide')) # [ 0.2         0.33333333 -0.42857143  0.5       ]
print(tf.math.mod(x, y, name='mod')) # [1 2 4 4]

# 음수, 부호 반환 
print('tf.neg=', tf.math.negative(x)) # [-1 -2  3 -4]
print('tf.sign=', tf.math.sign(x)) # [ 1  1 -1  1]

# 제곱/제곱근/거듭제곱 
print(tf.math.abs(x)) # [1 2 3 4]
print(tf.math.square(x)) # 제곱 - [ 1  4  9 16]
print(tf.math.sqrt([4.0, 9.0, 6.0])) # 제곱근
print(tf.math.pow(x, 3)) # 거듭제곱-[  1   8 -27  64]

# 지수와 로그 
print('e=', tf.math.exp(1.0).numpy()) # e= 2.7182817
print(tf.math.exp(2.0)) 
print(tf.math.log(8.0)) # 밑수e 자연로그



########################################
## 지수 함수를 이용한 sigmoid 활성함수 
########################################
 
import numpy as np 

def sigmoid(x) : # x : 입력변수 
    ex = tf.math.exp(-x)
    y = 1 / (1 + ex)
    return y # y : 출력변수(예측치)


x_data = np.array([1.0, 2.0, 5.0]) # 입력자료


# 함수 호출 
y1 = sigmoid(x_data)
print(y1.numpy())# [0.73105858 0.88079708 0.99330715] : 확률 


########################################
## 지수 함수를 이용한 softmax 활성함수 
########################################
    
# 함수 정의  
def softmax(x) :    
    ex = tf.math.exp(x - x.max())
    y = ex / sum(ex.numpy())
    return y


# 함수 호출 
y2 = softmax(x_data)
print(y2.numpy())# [0.01714783 0.04661262 0.93623955] : 확률 

###########################################################################################################

'''
1. Tensor 모양변경  
 - tf.transpose : 전치행렬 
 - tf.reshape : 모양 변경 
'''

import tensorflow as tf

x = tf.random.normal([2, 3]) # 정규분포 난수 생성 
print(x)

xt = tf.transpose(x)
print(xt)

x_r = tf.reshape(tensor=x, shape=[1, 6]) # (tensor, shape)
print(x_r)


'''
2. squeeze
 - 차원의 size가 1인 경우 제거
'''

t = tf.zeros( (1,2,1,3) )
t.shape # [1, 2, 1, 3]

print(tf.squeeze(t)) # shape=(2, 3)

print(tf.squeeze(t).shape) # (2, 3)

print(tf.squeeze(t).get_shape()) # (2, 3)


'''
3. expand_dims
 - tensor에 축 단위로 차원을 추가하는 함수 
'''

const = tf.constant([1,2,3,4,5]) # 1차원 

print(const)
print(const.shape) # (5,)

d0 = tf.expand_dims(const, axis=0) # 행축 2차원 
print(d0) 
    
d1 = tf.expand_dims(const, axis=1) # 열축 2차원 
print(d1)
