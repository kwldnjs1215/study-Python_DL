import tensorflow as tf # ver2.x

##############################
## 차원축소 통계/수학 함수
##############################
'''
 tf.reduce_sum(input_tensor, axis) : 지정한 차원을 대상으로 원소들 덧셈
 tf.reduce_mean(input_tensor, axis) : 지정한 차원을 대상으로 원소들 평균
 tf.reduce_prod(input_tensor, axis) : 지정한 차원을 대상으로 원소들 곱셈
 tf.reduce_min(input_tensor, axis) : 지정한 차원을 대상으로 최솟값 계산
 tf.reduce_max(input_tensor, axis) : 지정한 차원을 대상으로 최댓값 계산

 tf.reduce_all(input_tensor) : tensor 원소가 전부 True -> True 반환
 tf.reduce_any(input_tensro) : tensor 원소가 하나라도 True -> True 반환  
'''
data = [[1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]
data
print(tf.reduce_sum(data, axis=0)) # 행축 합계:열 단위 합계 
print(tf.reduce_sum(data, axis=1)) # 열축 합계:행 단위 합계  

# 전체 data 연산 
print(tf.reduce_mean(data)) # 전체 data = 2.5  
print(tf.reduce_mean(data, axis=0)) # [2.5 2.5]
print(tf.reduce_max(data)) # 3.5 
print(tf.reduce_min(data)) # 1.5

bool_data = [[True, True], [False, False]] 
print(tf.reduce_all(bool_data)) # False
print(tf.reduce_any(bool_data)) # True

########################################################################################################

#######################
##    난수 생성 함수 
#######################
''' 
tf.random.normal(shape, mean, stddev)  : 평균,표준편차 정규분포
tf.truncated.normal(shape, mean, stddev) : 표준편차의 2배 수보다 큰 값은 제거하여 정규분포 생성 
tf.random.uniform(shape, minval, maxval) : 균등분포 난수 생성
tf.random.shuffle(value) : 첫 번째 차원 기준으로 텐서의 원소 섞기
tf.random.set_seed(seed)  : 난수 seed값 설정 
'''

import tensorflow as tf # ver2.x
import matplotlib.pyplot as plt # 시각화 도구 

# 시각화 오류 해결방법 : Fatal Python error: Aborted  
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



# 표준정규분포를 따르는 난수 생성(2행3열)  
norm = tf.random.normal([2,3], mean=0, stddev=1) 
print(norm) # 객체 보기 

uniform = tf.random.uniform([2,3], minval=0, maxval=1) 
print(uniform) # 객체 보기 

matrix = [[1,2], [3,4], [5,6]] # 중첩list : (3, 2)
shuff = tf.random.shuffle(matrix) 
print(shuff) 

# seed값 지정 : 동일한 난수 생성   
tf.random.set_seed(1234)
a = tf.random.uniform([1]) 
b = tf.random.normal([1])  

print('a=',a.numpy())  
print('b=',b.numpy())  

# 제공 함수 확인 
dir(tf.random)


####################################
# 정규분포, 균등분포 차트 시각화
####################################

# 정규분포(평균:0, 표준편차:2) 
norm = tf.random.normal([1000], mean=175, stddev=5.5) 
norm
data = norm.numpy() # data 추출 
plt.hist(data) 
plt.show()
 
# 균등분포(0~1) 
uniform = tf.random.uniform([1000], minval=2.5, maxval=5.5) 
data2 = uniform.numpy() # data 추출
plt.hist(data2) 
plt.show() 

########################################################################################################

'''
선형대수 연산 함수  
  단위행렬 -> tf.linalg.eye(dim) 
  정방행렬의 대각행렬 -> tf.linalg.diag(x)  
  정방행렬의 행렬식 -> tf.linalg.det(x)
  정방행렬의 역행렬 -> tf.linalg.inv(x)
  두 텐서의 행렬곱 -> tf.linalg.matmul(x, y)
'''

import tensorflow as tf
import numpy as np

# 정방행렬 데이터 생성 
x = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 
y = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 


eye = tf.linalg.eye(2) # 단위행렬
print(eye.numpy()) 
 

dia = tf.linalg.diag(x) # 대각행렬 
mat_deter = tf.linalg.det(x) # 정방행렬의 행렬식  
mat_inver = tf.linalg.inv(x) # 정방행렬의 역행렬
mat = tf.linalg.matmul(x, y) # 행렬곱 반환 

print(x)
print(dia.numpy()) 
print(mat_deter.numpy())
print(mat_inver.numpy())
print(mat.numpy())


## 행렬곱 
A = tf.constant([[1,2,3], [3,4,2], [3,2,5]]) # A행렬 
B = tf.constant([[15,3, 5], [3, 4, 2]]) # B행렬  

A.get_shape() # [3, 3]
B.get_shape() # [2, 3]

# 행렬곱 연산 
mat_mul = tf.linalg.matmul(a=A, b=B)
print(mat_mul.numpy())

########################################################################################################

"""
퍼셉트론 구현 : ppt.3 참고  
 - 가중치(weight) : X 변수 중요도 조절변수
 - 편향(bais) : 뉴런의 활성화 조절변수(기울기 조절)
"""

import numpy as np 


# 1. 신경망 조절변수(W, b)  
def init_variable() :
    variable = {} # dict
    variable['W'] = np.array([[0.1], [0.3]]) # 가중치   
    variable['b'] = 0.1 # 편향       
    return variable
    


# 2. 활성함수 : 항등함수
def activation(model) :
    return model 
 

    
# 3. 순방향(forward) 실행 
def forward(variable, X) :  # (조절변수, X변수) 
    W = variable['W'] # 가중치   
    b = variable['b'] # 편향    
    model = np.dot(X, W) + b # 망의총합
    y = activation(model) # 활성함수 
    return y



# 프로그램 시작점 
variable = init_variable() # 조절변수(W, b) 생성 
variable 
'''
{'W': array([[0.1],
        [0.3]]),
 'b': 0.1}
'''

X = np.array([[1.0, 0.5]]) # X변수
X # [[1. , 0.5]]
y = forward(variable, X) # 순방향 연산 

print('y_pred =', y) # y_pred = [[0.35]]

########################################################################################################

"""
단순선형회귀방정식(formula) 작성 : ppt.5 참고 
  y_pred = (X * w) + base
"""

import tensorflow as tf  # ver 2.x

# X, y 변수 : 상수 정의  
X = tf.constant(6.5) # 독립변수(1개) 
y = tf.constant(5.2) # 종속변수(1개) 

# w, b 변수 : 변수 정의 
w = tf.Variable(0.5) # 가중치(weight) 
b = tf.Variable(1.5) # 편향(base) 


# 회귀모델 : y예측치 반환 
def linear_model(X) : 
    global w, b # 전역변수 
    # y_pred = X * w + b 
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식  
    return y_pred 

# model 오차 : y - y_pred
def model_err(X, y) : 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # err = y - y_pred 
    return err 


# 손실 함수 : MSE 
def loss_fn(X, y) :
    err = model_err(X, y) 
    loss = tf.reduce_mean(tf.square(err)) # 손실함수 
    return loss


# 프로그램 시작점 
print('-'*40)
print('<<가중치, 편향 초기값>>')    
print('가중치(w) =', w.numpy(), '편향(b) =', b.numpy())
print('-'*40)

print('y_pred =', linear_model(X).numpy()) # X = 6.5
print('y =', y.numpy())
print('model error = %.5f'%(model_err(X, y)))    
print('loss value = %.5f'%(loss_fn(X, y)))

'''
----------------------------------------
<<가중치, 편향 초기값>>
가중치(w) = 0.5 편향(b) = 1.5
----------------------------------------
y_pred = 4.75
y = 5.2
model error = 0.45000
loss value = 0.20250
'''

# 가중치, 편향 수정 
w.assign(0.6) # 가중치 수정 
b.assign(1.2) # 편향 수정 


print('-'*40)
print('<<가중치, 편향 수정>>')    
print('가중치(w) =', w.numpy(), '편향(b) =', b.numpy())
print('-'*40)

print('y_pred =', linear_model(X).numpy()) # X = 6.5
print('y =', y.numpy())
print('model error = %.5f'%(model_err(X, y)))    
print('loss value = %.5f'%(loss_fn(X, y)))
'''
----------------------------------------
<<가중치, 편향 수정>>
가중치(w) = 0.6 편향(b) = 1.2
----------------------------------------
y_pred = 5.1000004
y = 5.2
model error = 0.10000
loss value = 0.01000
'''
# [해설] 
# 조절변수(w, b) 수정으로 손실값이 낮아짐
# 딥러닝 최적화 알고리즘 : 최적의 가중치와 편형으로 수정(update)

########################################################################################################

"""
다중선형회귀방정식 작성 
  예) 독립변수 2개 
   y_pred = (X1 * w1 + X2 * w2) + base
"""

import tensorflow as tf 

# X, y변수 정의 
X = tf.constant([[1.0, 2.0]]) # 독립변수(입력)
X.shape # [1, 2]
y = tf.constant(2.5)  # 종속변수(정답)

# w, b변수 정의 : 초기값 난수   
tf.random.set_seed(1) # 난수 seed값 
w = tf.Variable(tf.random.normal([2, 1])) # 2개 난수 
b = tf.Variable(tf.random.normal([1])) # 1개 난수 
w.shape # [2, 1]
w.numpy()
'''
[[-1.1012203], -> w1
 [ 1.5457517]] -> w2
'''
b # 0.40308788

# 선형회귀모델  
def linear_model(X) : 
    global w, b
    # y_pred = X @ w + b
    y_pred = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return y_pred 

'''
행렬곱 
tf.linalg.matmul(X, w)
1. X, w 모두 행렬 
2. 수 일치 : X(1,2) vs w(2,1)
3. 자료형 일치 : 실수형 vs 실수형    
'''

# 모델 오차(model error) 
def model_err(X, y) : 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 오차  
    return err 


# 손실 함수(loss function)
def loss_fn(X, y) :
    err = model_err(X, y) # 오차 
    loss = tf.reduce_mean(tf.square(err)) # 손실함수   
    return loss


# 프로그램 시작점 
print('-'*30)
print('<<가중치, 편향 초기값>>')
print('가중치(w) : \n', w.numpy(), '\n편향(b) :', b.numpy())
print('-'*30)

# 모델 오차 
err = model_err(X, y)
print('err =', err.numpy())
# err = [[0.10662889]]

# 손실/비용 함수
loss = loss_fn(X, y)
print('손실(loss) =', loss.numpy())
# 손실(loss) = 0.011369721

########################################################################################################

"""
딥러닝 최적화 알고리즘 이용 단순선형회귀모델 
"""

import tensorflow as tf  # tensorflow 도구 
import matplotlib.pyplot as plt # 회귀선 시각화 

# 시각화 오류 해결방법 : Fatal Python error: Aborted  
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 1. X, y변수 생성 
X = tf.constant([1, 2, 3], dtype=tf.float32) # 독립변수(입력) 
y = tf.constant([2, 4, 6], dtype=tf.float32) # 종속변수(정답) 
X.shape # [3]

# 2. w, b변수 정의 
tf.random.set_seed(123)
w  = tf.Variable(tf.random.normal([1])) # 가중치 : 난수 
b  = tf.Variable(tf.random.normal([1])) # 편향 : 난수 


# 3. 회귀모델 
def linear_model(X) :  
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식 
    return y_pred 


# 4. 손실/비용 함수(loss/cost function) : 손실반환(MSE)
def loss_fn() : # 인수 없음 
    global X, y 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 오차 = 정답 - 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE  
    return loss # 손실 


# 5. model 최적화 객체  
optimizer = tf.optimizers.SGD(learning_rate=0.1) # 딥러닝 최적화 알고리즘

dir(tf.optimizers)
'''
Adam : 최신 버전 
SGD : 확률적 경사하강법 
'''

# 6. 반복학습 : 100번 
for step in range(100) :
    # 조절변수 수정(update) -> 손실(loss) 최소화 
    optimizer.minimize(loss=loss_fn, var_list=[w, b]) # (손실값, 조절변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())

    # a, b 변수 update 
    print(f'가중치(w) = {w.numpy()}, 편향(b) = {b.numpy()}')

'''
SGD(learning_rate=0.01)

step = 100 , loss value = 0.1620378
가중치(w) = [1.5324576], 편향(b) = [1.0627847]

SGD(learning_rate=0.1)
step = 100 , loss value = 0.002018992
가중치(w) = [1.9478129], 편향(b) = [0.11863367]
'''

# 7. 최적화된 model 검증 
X_test = [2.5] # test set 

y_pred = linear_model(X_test)

print(y_pred.numpy()) # [4.988166]

# 전체 dataset 
y_pred = linear_model(X) # [1,2,3]
print(y_pred.numpy()) # [2.0664465 4.0142593 5.9620724]

print(y.numpy()) # [2. 4. 6.]


# 회귀선 시각화 
plt.plot(X.numpy(), y.numpy(), 'bo') # 산점도 
plt.plot(X.numpy(), y_pred.numpy(), 'r-') # 회귀선 
plt.show()

########################################################################################################

"""
딥러닝 최적화 알고리즘 이용 단순선형회귀모델 + csv file 
"""

import tensorflow as tf # 최적화 알고리즘 
import pandas as pd  # csv file 
from sklearn.preprocessing import minmax_scale # 정규화 
from sklearn.metrics import mean_squared_error # model 평가 

iris = pd.read_csv('C:/ITWILL/7_Tensorflow/data/iris.csv')
print(iris.info())

# 1. X, y data 생성
x_data = iris['Sepal.Length'] # 독립변수 
y_data = iris['Petal.Length'] # 종속변수 

# float64 -> float32

# 2. X, y변수 만들기     
X = tf.constant(x_data, dtype=tf.float32) # dtype 지정 
y = tf.constant(y_data, dtype=tf.float32) # dtype 지정 


# 3. a,b 변수 정의 : 초기값 - 난수  
tf.random.set_seed(123)
w = tf.Variable(tf.random.normal([1])) # 가중치 : float32
b = tf.Variable(tf.random.normal([1])) # 편향 : float32


# 4. 회귀모델 
def linear_model(X) : # 입력 : X -> y예측치 
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식 
    return y_pred 


# 5. 손실/비용 함수(loss/cost function) : 손실반환(MSE)
def loss_fn() : # 인수 없음 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 정답 - 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE  
    return loss


# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체 
optimizer = tf.optimizers.Adam(learning_rate=0.5) # lr : 0.9 ~ 0.0001
# learning rate. Defaults to 0.001

print(f'기울기(w) 초기값 = {w.numpy()}, 절편(b) 초기값 = {b.numpy()}')

# 7. 반복학습 : 300회
for step in range(300) : # 100 -> 300 
    optimizer.minimize(loss=loss_fn, var_list=[w, b])#(손실값, 조절변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())
    # a, b 변수 update 
    print(f'기울기(w) = {w.numpy()}, 절편(b) = {b.numpy()}')
    
'''
Adam(learning_rate=0.5), 반복학습 : 100회
step = 100 , loss value = 1.4621323
기울기(w) = [0.8448554], 절편(b) = [-1.039748]

Adam(learning_rate=0.5), 반복학습 : 300회
step = 300 , loss value = 0.7697913
기울기(w) = [1.661757], 절편(b) = [-5.9326158]
'''

# 8.  최적화된 model 평가 
y_pred = linear_model(X.numpy())
mse = mean_squared_error(y_true=y.numpy(), y_pred=y_pred.numpy())

print('MSE=',mse) # MSE= 0.76979125

########################################################################################################

"""
딥러닝 최적화 알고리즘 이용 다중선형회귀모델 + csv file 
 - y변수 : 1칼럼, X변수 : 2~4칼럼 
 - 딥러닝 최적화 알고리즘 : Adam 적용 
 - 반복학습(step) 적용 
"""

import tensorflow as tf  # 딥러닝 최적화 알고리즘
from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import mean_squared_error # 평가 
from sklearn.preprocessing import minmax_scale # 정규화(0~1)

# 1. dataset load 
X, y = load_iris(return_X_y=True)
X.shape # (150, 4)

# X변수 정규화 
X_nor = minmax_scale(X)
type(X_nor) # numpy.ndarray

X_nor.dtype # dtype('float64')
dir(X_nor) # astype

# float64 -> float32
X_nor = X_nor.astype('float32')

# y변수 : 1칼럼, X변수 : 2~4칼럼 
y_data = X_nor[:,0] # 1칼럼  - y변수 
x_data = X_nor[:,1:] # 2~4칼럼 - x변수 
y_data.shape # (150,)
x_data.shape # (150, 3)


# 2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=123)



# 3. w, b 변수 정의 : update 대상 
tf.random.set_seed(123) # w,b 난수 seed값 지정 
w = tf.Variable(tf.random.normal(shape=[3, 1])) # float32
b = tf.Variable(tf.random.normal(shape=[1])) # float32


# 4. 회귀모델 정의 : 행렬곱 이용 
def linear_model(X) : # X:입력 -> y 예측치 : 출력 
    y_pred = tf.linalg.matmul(X, w) + b 
    return y_pred 

'''
InvalidArgumentError: cannot compute MatMul as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:MatMul]
'''
# 5. 손실/비용 함수 정의
def loss_fn() : # 인수 없음 
    y_pred = linear_model(x_train) # y 예측치
    err = tf.math.subtract(y_train, y_pred) # y - y_pred 
    loss = tf.reduce_mean(tf.square(err)) # MSE 
    return loss 


# 6. 최적화 객체 생성 
opt = tf.optimizers.Adam(learning_rate=0.01) # 학습률 

print('초기값 : w =', w.numpy(), ", b =", b.numpy())


# 7. 반복학습
loss_value = [] # 손실 저장 

for step in range(100) : 
    opt.minimize(loss=loss_fn, var_list=[w, b]) 
    
    loss = loss_fn().numpy()
    loss_value.append(loss)
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ', loss value =', loss_fn().numpy())
        
        print('수정 : w =', w.numpy(), ", b =", b.numpy())
        
        
# 8. model 평가 : test 이용 
y_pred = linear_model(x_test)
y_pred    

mse = mean_squared_error(y_test, y_pred.numpy())
print('MSE =',mse) # MSE = 0.25900778


# loss vs step 
import matplotlib.pyplot as plt 

plt.plot(loss_value)
plt.ylabel('loss value')
plt.xlabel('epochs')
plt.show()


########################################################################################################

'''
Hyper parameter : 사용자가 지정하는 파라미터
 - learning rate : model 학습율(0.9 ~ 0.0001)
 - iteration size : model 반복학습 횟수(epoch)
 - batch size : model 공급 데이터 크기  
'''

import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.datasets import load_iris


iris = load_iris() # 0-1에 근사한 변수 선택
X = iris.data

x_data = X[:, 2] # 꽃잎 길이(3)
y_data = X[:, 3] # 꽃잎 넓이(4)


# Hyper parameter
learning_rate = 0.0001 # 학습율 
iter_size = 1000 # 학습횟수 
'''
1차 테스트 : lr = 0.001, iter size = 100 -> 안정적인 형태 
2차 테스트 : lr = 0.05, iter size = 100 -> 최소점 수렴속도 빠름 
3차 테스트 : lr = 0.0001, iter size = 1000 -> 최소점 수렴속도 느름,학습횟수 증가   
'''

X = tf.constant(x_data, dtype=tf.float32) 
Y = tf.constant(y_data, dtype=tf.float32) 


tf.random.set_seed(123)
w = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))


# 4. 회귀모델 
def linear_model(X) : # 입력 X
    y_pred = tf.multiply(X, w) + b # y_pred = X * w + b
    return y_pred

# 5. 비용 함수 정의 
def loss_fn_l1() : # MAE : L1 loss function : Lasso 회귀  
    y_pred = linear_model(X) # 예측치 : 회귀방정식  
    err = Y - y_pred # 오차 
    loss = tf.reduce_mean(tf.abs(err)) 
    return loss

def loss_fn_l2() : # MSE : L2 loss function : Lidge 회귀  
    y_pred = linear_model(X) # 예측치 : 회귀방정식  
    err = Y - y_pred # 오차 
    loss = tf.reduce_mean(tf.square(err)) 
    return loss

# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체  
optimizer = tf.optimizers.SGD(lr = learning_rate) 


loss_l1_val = [] # L1 cost value
loss_l2_val = [] # L2 cost value


# 7. 반복학습 : 100회 
for step in range(iter_size) : 
    # 오차제곱평균 최적화 : 손실값 최소화 -> [w, b] 갱신(update)
    optimizer.minimize(loss_fn_l1, var_list=[w, b])#(손실값, 수정 대상)
    optimizer.minimize(loss_fn_l2, var_list=[w, b])#(손실값, 수정 대상)
    
    # loss value save
    loss_l1_val.append(loss_fn_l1().numpy())
    loss_l2_val.append(loss_fn_l2().numpy())
    
       
       
##### 최적화된 model(L1 vs L2) 비교 #####
''' loss values '''
print('loss values')
print('L1 =', loss_l1_val[-5:])
print('L2 =', loss_l2_val[-5:])

'''L1,L2 loss, learning rate, iteration '''
plt.plot(loss_l1_val, '-', label='loss L1')
plt.plot(loss_l2_val, '--', label='loss L2')
plt.title('L1 loss vs L2 loss')
plt.xlabel('Generation')
plt.ylabel('Loss values')
plt.legend(loc='best')
plt.show()

