'''
index 리턴 
  1. argmin/argmax
   - 최소/최대 값의 index 반환 
  2. argsort
   - 정렬 후 index 반환
'''
import tensorflow as tf # ver2.x

a = tf.constant([5,2,1,4,3], dtype=tf.int32) # 1차원 
b = tf.constant([4,5,2,3,1]) # 1차원 
c = tf.constant([[5,4,2], [3,2,4]]) # 2차원 
'''
array([[5, 4, 2],
       [3, 2, 4]])>
'''

# 1. argmin/argmax : 최솟값/최댓값 색인반환 
# 1차원 : argmin/argmax(input)
print(tf.argmin(a).numpy()) # 2
print(tf.argmax(b).numpy()) # 1

# 2차원 : argmin/argmax(input, axis=0) 
print(tf.argmin(c, axis=0).numpy()) # 행축(열 단위)
print(tf.argmin(c, axis=1).numpy()) # 열축(행 단위)

print(tf.argmax(c, axis=0).numpy()) # 행축(열 단위)
print(tf.argmax(c, axis=1).numpy()) # 열축(행 단위)


# 2. argsort : 오름차순정렬 후 색인반환 
# 형식) tf.argsort(values, direction='ASCENDING')
print(tf.argsort(a).numpy()) # [2 1 4 3 0] : [5,2,1,4,3]
print(tf.argsort(b).numpy()) # [4 2 3 0 1] : [4,5,2,3,1]

# 내림차순정렬 -> 색인 반환 
print(tf.argsort(a, direction='DESCENDING').numpy()) # [0 3 4 1 2]

########################################################################################

'''
활성함수(activation function)
 - model의 결과를 출력 y로 활성화 시키는 비선형 함수 
 - 유형 : sigmoid, softmax 
'''

import tensorflow as tf
import numpy as np

### 1. sigmoid function : 이항분류
def sigmoid_fn(x) : # x : 입력변수 
    ex = tf.math.exp(-x)   
    y = 1 / (1 + ex)
    return y # y : 출력변수(예측치)    


for x in np.arange(-5.0, 6.0) : # -5 ~ +5
    y = sigmoid_fn(x)  
    print(f"x : {x} -> y : {y.numpy()}")
        
    
### 2. softmax function : 다항분류
def softmax_fn(x) :    
    ex = tf.math.exp(x - x.max())
    #print(ex.numpy())
    y = ex / ex.numpy().sum()
    return y


x_data = np.arange(1.0, 6.0) # 1~5

y = softmax_fn(x_data)  
print(y.numpy())
print(y.numpy().sum()) # 1.0
    
########################################################################################

"""
엔트로피(Entropy) 
 - 확률변수 p에 대한 불확실성의 측정 지수 
 - 값이 클 수록 일정한 방향성과 규칙성이 없는 무질서(chaos) 의미
 - Entropy = -sum(p * log(p))
"""

import numpy as np

# 1. 불확실성이 큰 경우(p1: 앞면, p2: 뒷면)
p1 = 0.5; p2 = 0.5

entropy = -sum([p1 * np.log2(p1), p2 * np.log2(p2)])  
print('entropy =', entropy) # entropy = 1.0


# 2. 불확실성이 작은 경우(x1: 앞면, x2: 뒷면) 
p1 = 0.9; p2 = 0.1

entropy2 = -sum([p1 * np.log2(p1), p2 * np.log2(p2)])
print('entropy2 =', entropy2) # entropy = 0.468995


'''
Cross Entropy    
  - 두 확률변수 x와 y가 있을 때 x를 관찰한 후 y에 대한 불확실성 측정
  - Cross 의미 :  y=1, y=0 일때 서로 교차하여 손실 계산 
  - 식 = -( y * log(y_pred) + (1-y) * log(1-y_pred))

  왼쪽 식 : y * log(y_pred) -> y=1 일 때 손실값 계산  
  오른쪽 식 : (1-y) * log(1-y_pred) -> y=0 일 때 손실값 계산 
'''

import tensorflow as tf 

y_preds = [0.02, 0.98] # model 예측값(0~1)

y = 1 # 정답
for y_pred in y_preds :
    loss_val = -(y * tf.math.log(y_pred)) # y=1 일때 손실값 
    print(loss_val.numpy())
'''
3.912023    : y=1 vs y_pred=0.02
0.020202687 : y=1 vs y_pred=0.98
'''

y = 0 # 정답
for y_pred in y_preds :
    loss_val = -((1-y) * tf.math.log(1-y_pred)) # y=0 일때 손실값 
    print(loss_val.numpy())
    
'''
0.020202687 : y=0 vs y_pred=0.02
3.912023    : y=0 vs y_pred=0.98
'''    

# Cross Entropy
y = 0 # y=1 or y=0 : 정답    
for y_pred in y_preds :    
    loss_val = -( y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred)) 
    print(loss_val.numpy())
    
'''
y = 1
3.912023    : y=1 vs 0.02
0.020202687 : y=1 vs 0.98

y = 0
0.020202687 : y=0 vs 0.02
3.912023    : y=0 vs 0.98
'''    
    
########################################################################################

"""
이항분류기 : 테스트 데이터 적용
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score #  model 평가 

# 1. x, y 공급 data 
# x변수 : [hours, video]
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # [6, 2]

# y변수 : [fail or pass] one-hot encoding 
y_data = [[1,0], [1,0], [1,0], [0,1], [0,1], [0,1]] # [6, 2] : 이항분류 


# 2. X, Y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) # shape=(6, 2)
y = tf.constant(y_data, tf.float32) # shape=(6, 2)


# 3. w, b변수 정의 : 초기값(난수)  
w = tf.Variable(tf.random.normal(shape=[2, 2])) # 가중치[입력수,출력수] 
b = tf.Variable(tf.random.normal(shape=[2])) # 편향[출력수] 


# 4. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return model 
    
# 5. sigmoid 함수  : 이항분류 활성함수 
def sigmoid_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.sigmoid(model) # 0~1확률 
    return y_pred 
    
# 6. 손실함수 : cross entropy 이용 
def loss_fn() : # 인수 없음 
    y_pred = sigmoid_fn(X)
    loss = -tf.reduce_mean(y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    return loss


# 7. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.5)


# 8. 반복학습 
for step in range(100) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
    

# 9. 최적화된 model 검증 
y_pred = sigmoid_fn(X) # sigmoid 함수 호출 
print(y_pred.numpy()) # 확률예측(0~1) -> 10진수  
'''
     0           1
[[0.99883336 0.00110541]
 [0.94494206 0.05410441]
 [0.93284863 0.06603013]
 [0.07507661 0.926921  ]
 [0.00555122 0.99473345]
 [0.0013153  0.99876744]]
'''

# 확률 -> 10진수 
y_pred = tf.argmax(y_pred.numpy(), axis=1).numpy()
y_pred # [0, 0, 0, 1, 1, 1]

print(y.numpy()) # one hot encoding(2진수) -> 10진수  
'''
[[1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [0. 1.]
 [0. 1.]]
'''

y_true = tf.argmax(y.numpy(), axis=1).numpy()
y_true # [0, 0, 0, 1, 1, 1]

# model 평가 
acc = accuracy_score(y_true, y_pred) # (10진수, 10진수)
print(acc) # 1.0

########################################################################################

"""
이항분류기 : 실제 데이터(iris) 적용 
"""

import tensorflow as tf
from sklearn.datasets import load_iris # dataset
from sklearn.preprocessing import minmax_scale # x변수 정규화 
from sklearn.preprocessing import OneHotEncoder # y변수 전처리
from sklearn.metrics import accuracy_score #  model 평가 


# 1. data load  
X, y = load_iris(return_X_y=True)
X.shape # (150, 4)
y # 0~2

# x변수 선택 
X = X[:100]

# y변수 선택 
y = y[:100]  


# 2.  X, y 전처리 
x_data = minmax_scale(X) # x변수 정규화 : 0~1

# y변수 : one-hot 인코딩 
y_data = OneHotEncoder().fit_transform(y.reshape([-1, 1])).toarray()
y_data
'''
0 -> [1., 0.] 
1 -> [0., 1.] 
'''

# 3. X, y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) 
y = tf.constant(y_data, tf.float32) 
X.shape # [100, 4]
y.shape # [100, 2]

# 4. w, b변수 정의 : 초기값(난수) 
w = tf.Variable(tf.random.normal(shape=[4, 2])) # w[입력수,출력수]
b = tf.Variable(tf.random.normal(shape=[2])) # b[출력수]


# 5. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b 
    return model 
    
# 6. sigmoid 함수   
def sigmoid_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.sigmoid(model) 
    return y_pred 
    
# 7. 손실함수  
def loss_fn() : # 인수 없음 
    y_pred = sigmoid_fn(X)
    # cross entropy : loss value 
    loss = -tf.reduce_mean(y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    return loss


# 8. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.1)


# 9. 반복학습 
for step in range(100) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
    

## 10. 최적화된 model 검증
sigmoid_fn(X)
 
# F/T -> 0/1
# tf.cast(확률예측 > 0.5, dtype)
y_pred = tf.cast(sigmoid_fn(X).numpy() > 0.5, dtype=tf.float32)
'''
[ True, False] -> [1., 0.]
[False,  True] -> [0., 1.]
'''
print(y.numpy())
'''
[1, 0]
[0, 1]
'''
# 분류정확도 
acc = accuracy_score(y, y_pred) # (2진수, 2진수)
print('accuracy =',acc) # accuracy = 1.0

########################################################################################

"""
다항분류기 : 테스트 데이터 적용 
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score #  model 평가 
import numpy as np 

# 1. x, y 공급 data 
# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1]]) # [6, 2]

# [기타, 포유류, 조류] : [6, 3] 
y_data = np.array([ # one hot encoding 
    [1, 0, 0],  # 기타[0]
    [0, 1, 0],  # 포유류[1]
    [0, 0, 1],  # 조류[2]
    [1, 0, 0],  # 기타[0]
    [1, 0, 0],  # 기타[0]
    [0, 0, 1]   # 조류[2]
])


# 2. X, Y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) # [관측치, 입력수] - [6, 2]
Y = tf.constant(y_data, tf.float32) # [관측치, 출력수] - [6, 3]


# 3. w, b변수 정의 : 초기값(난수) -> update 
w = tf.Variable(tf.random.normal(shape=[2, 3])) # [입력수, 출력수]
b = tf.Variable(tf.random.normal(shape=[3])) # [출력수]


# 4. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return model 


# 5. softmax 함수   
def softmax_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.softmax(model) # 다항분류 활성함수 
    return y_pred 


# 6. 손실함수 : 손실값 반환 
def loss_fn() : # 인수 없음 
    y_pred = softmax_fn(X)
    # cross entropy : loss value 
    loss = -tf.reduce_mean(Y * tf.math.log(y_pred) + (1-Y) * tf.math.log(1-y_pred))
    return loss


# 7. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.1)


# 8. 반복학습 
for step in range(100) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
  
    
# 9. 최적화된 모델 검증 
y_pred = softmax_fn(X).numpy()
print(y_pred) # 확률예측 
'''
[[9.7692287e-01 2.0690689e-02 2.3864591e-03] -> 0
 [1.4535640e-02 9.1981906e-01 6.5645322e-02] -> 1
 [1.3886066e-02 2.3965197e-02 9.6214873e-01] -> 2
 [9.7692287e-01 2.0690689e-02 2.3864591e-03] -> 0
 [9.6333873e-01 5.5645133e-04 3.6104888e-02] -> 0
 [1.3886066e-02 2.3965197e-02 9.6214873e-01]]-> 2
'''
y_pred = tf.argmax(y_pred, axis = 1)

print(Y.numpy())
'''
[[1. 0. 0.] -> 0
 [0. 1. 0.] -> 1
 [0. 0. 1.] -> 2
 [1. 0. 0.] -> 0
 [1. 0. 0.] -> 0
 [0. 0. 1.]]-> 2
'''

y_true = tf.argmax(Y.numpy(), axis = 1)


acc = accuracy_score(y_true, y_pred)
print(acc) #  1.0

########################################################################################

"""
다항분류기 : 실제 데이터(iris) 적용
"""

import tensorflow as tf # 최적화 알고리즘 
from sklearn.datasets import load_iris # datast 
from sklearn.preprocessing import minmax_scale # x변수 전처리
from sklearn.preprocessing import OneHotEncoder # y변수 전처리
from sklearn.metrics import accuracy_score #  model 평가 

tf.random.set_seed(123) # seed 고정 - 동일 결과 

# 1. data load  
X, y = load_iris(return_X_y=True)


# 2. X, y변수 전처리 
x_data = minmax_scale(X) # x변수 : 정규화 

# y변수 : one-hot 인코딩 
y_data = OneHotEncoder().fit_transform(y.reshape([-1, 1])).toarray()
'''
10진수 -> 2진수 
  0  -> [1, 0, 0]
  1  -> [0, 1, 0]
  2  -> [0, 0, 1]
'''

# 3. X, Y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) # [관측치, 입력수] - [150, 4]
y = tf.constant(y_data, tf.float32) # [관측치, 출력수] - [150, 3]


# 4. w, b변수 정의 : 초기값(난수) -> update 
w = tf.Variable(tf.random.normal(shape=[4, 3])) # [입력수, 출력수]
b = tf.Variable(tf.random.normal(shape=[3])) # [출력수]


# 5. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return model 


# 6. softmax 함수  : 다항분류 활성함수 
def softmax_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.softmax(model) # softmax + model
    return y_pred 


# 7. 손실함수 : 정답(Y) vs 예측치(y_pred) -> 손실값 반환 
def loss_fn() : # 인수 없음 
    y_pred = softmax_fn(X)
    # cross entropy : loss value 
    loss = -tf.reduce_mean(y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    return loss


# 8. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.1)


# 9. 반복학습 
for step in range(150) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
  
    
# 10. 최적화된 model 검증 
y_pred = softmax_fn(X).numpy() # 예측치 반환 

y_pred = tf.argmax(y_pred, axis = 1) # 10진수 
y_true = tf.argmax(y, axis = 1) # 10진수 

acc = accuracy_score(y_true, y_pred)
print('accuracy =',acc) 
# step=100 : accuracy = 0.9533333333333334
# step=150 : accuracy = 0.9733333333333334

