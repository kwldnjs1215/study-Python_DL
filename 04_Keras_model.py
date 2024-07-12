
"""
Keras : High Level API  
"""

# dataset 
from sklearn.datasets import load_iris # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

# keras model 
import tensorflow as tf
from tensorflow.keras import Sequential # keara model 
from tensorflow.keras.layers import Dense # DNN layer 
import numpy as np 
import random 


## karas 내부 weight seed 적용 
tf.random.set_seed(123) # global seed 
np.random.seed(123) # numpy seed
random.seed(123) # random seed 


# 1. dataset laod 
X, y = load_iris(return_X_y=True)


# 2. 공급 data 생성 : 훈련셋, 검증셋 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. DNN model 생성 
model = Sequential() 
dir(model)
'''
add() : 레이어 추가 
compile() : 학습과정 설정
fit() : 모델 학습 
summary() : 레이어 확인 
predict() : y예측치  
'''

# 4. DNN model layer 구축 

# hidden layer1 : 1층(w[4,12], b=12) 
model.add(Dense(units=12, input_shape=(4,), activation='relu'))# 1층 

# hidden layer2 : 2층(w[12,6], b=6) 
model.add(Dense(units=6, activation='relu'))# 2층

# output layer : 3층(w[6,1], b=1) 
model.add(Dense(units=1))# 3층 : 활성함수 없음 

# 레이어 확인 
model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param(w,b) #   
=================================================================
 dense (Dense)               (None, 12)                60=4*12+12        
                                                                 
 dense_1 (Dense)             (None, 6)                 78=12*6+6        
                                                                 
 dense_2 (Dense)             (None, 1)                 7=6*1+1         
                                                                 
=================================================================
Total params: 145
'''

# 5. model compile : 학습과정 설정 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
'''
optimizer : 딥러닝 최적화 알고리즘(sgd, adam)
loss : 손실함수(mse, cross_entropy)
metrics : 평가방법(mae, accuracy)   
'''

# 6. model training 
model.fit(x=x_train, y=y_train,  # 훈련셋 
          epochs=100,  # 반복학습 
          verbose=1,   # 학습과정 출력 
          validation_data=(x_val, y_val)) # 검증셋   
'''
Epoch 100/100
4/4 [==============================] - 0s 10ms/step 
- loss: 0.0586 - mae: 0.1818 - val_loss: 0.0596 - val_mae: 0.1663
'''
# 7. model testing 
y_pred = model.predict(x_val)
y_true = y_val

mse = mean_squared_error(y_true, y_pred)
print('mse=', mse) # mse= 0.05959518977080322

r2_score = r2_score(y_true, y_pred)
print('r2 score =', r2_score) # r2 score = 0.9233289331093542

####################################################################################################

"""
keras history 기능
 - model 학습과정과 검증과정의 손실(loss)을 기억하는 기능 
"""

# dataset 
from sklearn.datasets import load_iris # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

# keras model 
import tensorflow as tf
from tensorflow.keras import Sequential # keara model 
from tensorflow.keras.layers import Dense # DNN layer 
import numpy as np 
import random as rd 

# tensorflow version
print(tf.__version__) # 2.10.0
# keras version 
print(tf.keras.__version__) # 2.10.0

## karas 내부 weight seed 적용 
tf.random.set_seed(123) # global seed 
np.random.seed(123) # numpy seed
rd.seed(123) # random seed 

# 1. dataset laod 
X, y = load_iris(return_X_y=True)

y # 0~2


# 2. 공급 data 생성 : 훈련셋, 검증셋 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


x_train.shape # (105, 4)
x_train.dtype # dtype('float64')

y_train.shape # (105,)


# 3. DNN model 생성 
model = Sequential() # keras model 
print(model) # Sequential object

# DNN model layer 구축 
'''
model.add(Dense(unit수, input_shape, activation)) : hidden1
model.add(Dense(unit수, activation)) : hidden2 ~ hiddenn
'''

##########################################
### hidden layer 2개 : unit=12, unit=6
##########################################

# hidden layer1 : unit=12 -> w[4, 12]
model.add(Dense(units=12, input_shape=(4,), activation='relu')) # 1층 

# hidden layer2 : unit=6 -> w[12, 6]
model.add(Dense(units=6, activation='relu')) # 2층

# output layer : unit=1 -> w[6, 1]
model.add(Dense(units=1)) # 3층 

# model layer 확인 
print(model.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param(in*out)+b   
=================================================================
dense_6 (Dense)              (None, 12)                60=(4*12)+12        
_________________________________________________________________
dense_7 (Dense)              (None, 6)                 78=(12*6)+6        
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 7=(6*1)+1         
=================================================================
'''

# 4. model compile : 학습과정 설정 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# 5. model training : train(105) vs test(45)
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=100, # 반복학습 횟수 
          verbose=1,  # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋  


# 6. model evaluation : validation data 
loss_val, mae = model.evaluate(x_val, y_val)
print('loss value =', loss_val)
print('mae =', mae)


# 7. model history : epoch에 따른 model 평가  
dir(model_fit)

model_fit.history['loss'] # train loss 
model_fit.history['val_loss'] # val loss 

# 1) epoch vs loss
import matplotlib.pyplot as plt
plt.plot(model_fit.history['loss'], 'y', label='train loss value')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss value')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()

# 2) epoch vs mae
plt.plot(model_fit.history['mae'], 'y', label='train mae')
plt.plot(model_fit.history['val_mae'], 'r', label='val mae')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.legend(loc='best')
plt.show()

####################################################################################################

"""
Keras : DNN model 생성을 위한 고수준 API
 
Keras 이항분류기 
 - X변수 : minmax_scale(0~1)
 - y변수 : one hot encoding(2진수 인코딩)
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)

from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense, Input # DNN layer 구축 
from tensorflow.keras.models import Model # model 생성 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. dataset load & 전처리 
X, y = load_iris(return_X_y=True)


# X변수 : 정규화
X = minmax_scale(X[:100]) # 100개 선택 
X.shape # (100, 4)

# y변수 : 2진수(one hot encoding)
y = to_categorical(y[:100])
y.shape # (100, 2)

# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. keras layer & model 생성

#################################################
## 1. Sequential API 방식 : 초보자용
#################################################

model = Sequential()

# hidden layer1 
model.add(Dense(units=8, input_shape =(4, ), activation = 'relu')) # 1층 

# hidden layer2  
model.add(Dense(units=4, activation = 'relu')) # 2층 

# output layer 
model.add(Dense(units=2, activation = 'sigmoid')) # 3층 


#################################################
## 2. Functional API 방식 : 개발자용(엔지니어)
#################################################
# 순방향(forward) 레이어 구축 : Input -> Hidden -> Output

input_dim = 4 # input data 차원
output_dim = 2 # output data 차원

# 1) input layer
inputs = Input(shape=(input_dim,)) # Input 클래스 이용

# 2) hidden layer1
hidden1 = Dense(units=8, activation='relu')(inputs) # 1층

# 3) hidden layer2
hidden2 = Dense(units=4, activation='relu')(hidden1) # 2층

# 4) output layer
outputs = Dense(units=output_dim, activation ='sigmoid')(hidden2)# 3층 

# model 생성 
model = Model(inputs, outputs) # Model 클래스 이용

model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 4)]               0         
                                                                 
 dense_37 (Dense)            (None, 8)                 40        
                                                                 
 dense_38 (Dense)            (None, 4)                 36        
                                                                 
 dense_39 (Dense)            (None, 2)                 10        
                                                                 
=================================================================
Total params: 86
'''










# 4. model compile : 학습과정 설정(이항분류기)
model.compile(optimizer='adam', 
              loss = 'binary_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(70) vs val(30) 
model.fit(x=x_train, y=y_train, 
          epochs=25,  
          verbose=1,  
          validation_data=(x_val, y_val)) 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

####################################################################################################

"""
keras 다항분류기 
 - X변수 : minmax_scale(0~1)
 - y변수 : one hot encoding(2진수 인코딩)
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)
from sklearn.metrics import accuracy_score  # model 평가 

from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # DNN layer 구축 
from tensorflow.keras.models import load_model # model load 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. dataset load & 전처리 
X, y = load_iris(return_X_y=True)

X.shape # (150, 4)
y.shape # (150,)


# X변수 : 정규화(0~1)
X = minmax_scale(X) 

# y변수 : one hot encoding
y = to_categorical(y) 


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. keras layer & model 생성
model = Sequential()

# hidden layer1 
model.add(Dense(units=12, input_shape =(4, ), activation = 'relu')) # 1층 

# hidden layer2 
model.add(Dense(units=6, activation = 'relu')) # 2층 

# output layer
model.add(Dense(units=3, activation = 'softmax')) # 3층 : [수정]

model.summary()
'''
___________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_3 (Dense)             (None, 12)                60=w[4x12]+b[12]        
                                                                 
 dense_4 (Dense)             (None, 6)                 78=w[12x6]+b[6]        
                                                                 
 dense_5 (Dense)             (None, 3)                 21=w[6x3]+b[3]        
                                                                 
=================================================================
Total params: 159
'''


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])


# 5. model training : train(105) vs val(45) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=200, # 반복학습 : [수정]
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 7. model save & load 
dir(model)

model.save('keras_model_iris.h5') # HDF5 파일 형식 

new_model = load_model('keras_model_iris.h5')


# 8. 평가셋(test) & 모델 평가 
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=123)

y_pred = new_model.predict(x_test) # 확률예측 
# 확률예측 -> 10진수 변경 
y_pred = tf.argmax(y_pred, axis=1)

# 2진수 -> 10진수 변경
y_test = tf.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)  
print('accuracy =', acc) # accuracy = 0.9466666666666667

####################################################################################################

"""
keras 모델에서 학습률 적용   
 optimizer=Adam(learning_rate = 0.01)
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)
from sklearn.metrics import accuracy_score  # model 평가 

from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # DNN layer 구축 
from tensorflow.keras.models import load_model # model load 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. dataset load & 전처리 
X, y = load_iris(return_X_y=True)

X.shape # (150, 4)
y.shape # (150,)


# X변수 : 정규화(0~1)
X = minmax_scale(X) # 

# y변수 : one hot encoding
y_one = to_categorical(y)  
print(y_one)
y_one.shape #  (150, 3)
'''
[1, 0, 0] <- 0
[0, 1, 0] <- 1
[0, 0, 1] <- 2
'''

# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y_one, test_size=0.3, random_state=123)


# 3. keras layer & model 생성 
model = Sequential()


# hidden layer1 
model.add(Dense(units=12, input_shape =(4, ), activation = 'relu')) # 1층 

# hidden layer2 
model.add(Dense(units=6, activation = 'relu')) # 2층 

# output layer 
model.add(Dense(units=3, activation = 'softmax')) # 3층 


# 4. model compile : 학습과정 설정(다항분류기) 
from tensorflow.keras import optimizers # 딥러닝 최적화 알고리즘 
dir(optimizers)
'''
Adam
RMSprop
SGD
'''
model.compile(optimizer=optimizers.Adam(learning_rate=0.01), 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(105) vs val(45) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=200, # 반복학습 
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

'''
Adam(learning_rate=0.001)
loss: 0.1898 - accuracy: 0.9111

Adam(learning_rate=0.01)
loss: 0.0525 - accuracy: 0.9778
'''


# 7. model save & load : HDF5 파일 형식 
model.save('keras_model_iris.h5')

my_model = load_model('keras_model_iris.h5')
 
####################################################################################################

"""
1. Mnist dataset 다항분류기 
2. Full batch vs Mini batch 
"""

from tensorflow.keras.datasets import mnist # mnist load 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding 
from tensorflow.keras import Sequential # keras model 생성 
from tensorflow.keras.layers import Dense # DNN layer 구축 

################################
## keras 내부 w,b변수 seed 적용 
################################
import tensorflow as tf
import numpy as np 
import random as rd

tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset load 
(x_train, y_train), (x_val, y_val) = mnist.load_data() # (images, labels)

# images : X변수 
x_train.shape # (60000, 28, 28) - (size, h, w) : 2d 제공 
x_val.shape # (10000, 28, 28)

x_train[0] # 0~255
x_train.max() # 255

# labels : y변수 
y_train.shape # (60000,)
y_train[0] # 5


# 2. X,y변수 전처리 

# 1) X변수 : 정규화 & reshape(2d -> 1d)
x_train = x_train / 255. # 정규화 
x_val = x_val / 255.


# reshape(2d -> 1d)
x_train = x_train.reshape(-1, 784) # (60000, 28*28)
x_val = x_val.reshape(-1, 784) # (10000, 28*28)


# 2) y변수 : class(10진수) -> one-hot encoding(2진수)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)



# 3. keras layer & model 생성
model = Sequential()

x_train.shape # (60000, 784)

input_dim = (784, ) # 1d

# hidden layer1 : w[784, 128]
model.add(Dense(units=128, input_shape=input_dim, activation='relu'))# 1층 

# hidden layer2 : w[128, 64]
model.add(Dense(units=64, activation='relu'))# 2층 

# hidden layer3 : w[64, 32]
model.add(Dense(units=32, activation='relu'))# 3층

# output layer : w[32, 10]
model.add(Dense(units=10, activation='softmax'))# 4층

#  model layer 확인 
model.summary()


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# 5. model training : train(70) vs val(30)
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=10, # 1epoch : 전체 데이터셋을 1회 소진 -> 600,000장 이미지   
          batch_size=100, # 1회 모델 공급 크기(100*600=60,000) 
          verbose=1, # 출력여부 
          validation_data= (x_val, y_val)) # 검증셋

'''
Epoch 10/10
600/600 [==============================] - 1s 2ms/step 
- loss: 0.0237 - accuracy: 0.9920 - val_loss: 0.0784 - val_accuracy: 0.9785
'''

# 6. model evaluation : val dataset 
print('model evaluation')
model.evaluate(x=x_val, y=y_val) # 10,000장 이미지 


####################################################################################################

"""
Flatten layer : input data의 차원을 은닉층에 맞게 일치
  ex) 2차원 이미지(28, 28) -> 1차원(784) 
"""

from tensorflow.keras.datasets import mnist # mnist load 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding 
from tensorflow.keras import Sequential # keras model 생성 
from tensorflow.keras.layers import Dense, Flatten # DNN layer 구축 

################################
## keras 내부 w,b변수 seed 적용 
################################
import tensorflow as tf
import numpy as np 
import random as rd

tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset load 
(x_train, y_train), (x_val, y_val) = mnist.load_data() # (images, labels)

# images : X변수 
x_train.shape # (60000, 28, 28) - (size, h, w) : 2d 제공 
x_val.shape # (10000, 28, 28)

x_train[0] # 0~255
x_train.max() # 255

# labels : y변수 
y_train.shape # (60000,)
y_train[0] # 5


# 2. X,y변수 전처리 

# 1) X변수 : 정규화 
x_train = x_train / 255. # 정규화 
x_val = x_val / 255.



# 2) y변수 : one-hot encoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# 3. keras layer & model 생성
model = Sequential()

x_train.shape # (60000, 28, 28)
input_dim = (28, 28) # 2d

# flatten layer [추가] 
model.add(Flatten(input_shape = input_dim))  # 2d/3d -> 1d(784)


# hidden layer1 : w[784, 128]
model.add(Dense(units=128, activation='relu'))# 1층 

# hidden layer2 : w[128, 64]
model.add(Dense(units=64, activation='relu'))# 2층 

# hidden layer3 : w[64, 32]
model.add(Dense(units=32, activation='relu'))# 3층

# output layer : w[32, 10]
model.add(Dense(units=10, activation='softmax'))# 4층


#  model layer 확인 
model.summary()


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', # y : one-hot encoding
              metrics=['accuracy'])


# 5. model training : train(70) vs val(30)
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=10, # 반복학습 횟수 
          batch_size=100, # 1회 공급data 크기  
          verbose=1, # 출력여부 
          validation_data= (x_val, y_val)) # 검증셋


# 6. model evaluation : val dataset 
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

####################################################################################################

"""
History : 훈련과 검증과정에서 발생하는 손실값/평가 결과 기억 기능 
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # DNN layer 구축 
import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 


################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() # (images, labels)

x_train.shape # (60000, 28, 28) 
y_train.shape # (60000,)


# 2. x,y변수 전처리 

# 1) x변수 : 정규화 & reshape(3d -> 2d)
x_train = x_train / 255.
x_val = x_val / 255.


# 3d -> 2d : [수정]
x_train = x_train.reshape(-1, 784) # 28 * 28 = 784
x_train.shape # (60000, 784)

x_val = x_val.reshape(-1, 784)
x_val.shape # (10000, 784)


# 2) y변수 : one hot encoding 
y_train = to_categorical(y_train) 
y_val = to_categorical(y_val) 
y_train.shape # (60000, 10)



# 3. keras model &  layer 구축
model = Sequential()


input_shape = (784,) 

# hidden layer1 : [784, 128] -> [input, output]
model.add(Dense(units=128, input_shape = input_shape, activation = 'relu')) # 1층 

# hidden layer2 : [128, 64] -> [input, output]
model.add(Dense(units=64, activation = 'relu')) # 2층 

# hidden layer3 : [64, 32] -> [input, output]
model.add(Dense(units=32, activation = 'relu')) # 3층

# output layer : [32, 10] -> [input, output]
model.add(Dense(units=10, activation = 'softmax')) # 4층 


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', # default=0.001
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(60,000) vs val(10,000) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=15, # 반복학습 
          batch_size = 100, # mini batch
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 7. model history 
print(model_fit.history.keys())


# loss vs val_loss 
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy 
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

####################################################################################################

"""
가중치 규제(Weight regularizer 
가중치가 너무 커지는 것을 방지하기 위해서 가중치를 감소시켜 훈련 데이터에 
과적합이 발생하지 않도록 하는 기법

형식)  
model.add(Dense(kernel_regularizer=regularizers.L1(규제인자))) : L1 방식 
model.add(Dense(kernel_regularizer=regularizers.L2(규제인자))) : L2 방식
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # DNN layer 구축 
from tensorflow.keras import regularizers # 가중치 규제

import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 
import time # 실행 시간 측정 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() # (images, labels)

x_train.shape # (60000, 28, 28) - 3d(size, h, w) -> 2d(size, h*w)
y_train.shape # (60000,)

x_train[0] # first image pixel 
x_train[0].max() # 0~255

plt.imshow(x_train[0])
plt.show()

y_train[0] # first label - 10진수 : 5


# 2. x,y변수 전처리 

# 1) x변수 : 정규화 & reshape(3d -> 2d)
x_train = x_train / 255.
x_val = x_val / 255.

x_train[0]


# 3d -> 2d : [수정]
x_train = x_train.reshape(-1, 784)
# 28 * 28 = 784
x_train.shape # (60000, 784)

x_val = x_val.reshape(-1, 784)
x_val.shape # (10000, 784)


# 2) y변수 : one hot encoding 
y_train = to_categorical(y_train) 
y_val = to_categorical(y_val) 
y_train.shape # (60000, 10)

y_train[0] # [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] <- 5

y_val[0] # [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.] <- 7


chktime = time.time() # 소요 시간 체크 

# 3. keras model 
model = Sequential()


# 4. DNN model layer 구축 

input_shape = (784,) 

'''
가중치 규제 적용 전  
kernel_regularizer=regularizers.L1(0.01) # L1방식 
kernel_regularizer=regularizers.L2(0.01) # L2방식 
'''

# hidden layer1 : [784, 128] -> [input, output]
model.add(Dense(units=128, input_shape = input_shape, 
                activation = 'relu', 
                kernel_regularizer=regularizers.L2(0.001))) # 1층 

# hidden layer2 : [128, 64] -> [input, output]
model.add(Dense(units=64, activation = 'relu',
                kernel_regularizer=regularizers.L2(0.01))) # 2층

# hidden layer3 : [64, 32] -> [input, output]
model.add(Dense(units=32, activation = 'relu')) # 3층

# output layer : [32, 10] -> [input, output]
model.add(Dense(units=10, activation = 'softmax')) # 4층 

# model layer 확인 
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 128)               100480=(784*128)+128    
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256=(128*64)+64      
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080=(64*32)+32      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                330=(32*10)+10       
=================================================================
Total params: 111,146
Trainable params: 111,146
Non-trainable params: 0
_________________________________________________________________
'''

# 5. model compile : 학습과정 설정(다항분류기)
model.compile(optimizer='adam', # default=0.001
              loss = 'categorical_crossentropy', # y : one hot encoding 
              metrics=['accuracy'])


# 6. model training : train(60,000) vs val(10,000) [수정]
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=15, # 반복학습 
          batch_size = 100, # 100*600 = 60000(1epoch)*10 = 600,000 -> mini batch
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 

chktime = time.time() - chktime  

print('실행 시간 : ', chktime)

# 7. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 8. model history 
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


import matplotlib.pyplot as plt 


# loss vs val_loss : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy : : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()


####################################################################################################

"""
Dropout : 무작위 네트워크 삭제 -> 과적합 최소화 

형식) model.add(Dropout(rate = 비율))
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense, Dropout # DNN layer 구축 
import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() # (images, labels)

x_train.shape # (60000, 28, 28) 
y_train.shape # (60000,)



# 2. x,y변수 전처리 

# 1) x변수 : 정규화 & reshape(3d -> 2d)
x_train = x_train / 255.
x_val = x_val / 255.


# 3d -> 2d : [수정]
x_train = x_train.reshape(-1, 784) # 28 * 28 = 784
x_train.shape # (60000, 784)

x_val = x_val.reshape(-1, 784)


# 2) y변수 : one hot encoding 
y_train = to_categorical(y_train) 
y_val = to_categorical(y_val) 



# 3. keras model & layer 구축
model = Sequential()


input_shape = (784,) 

'''
Dropout 적용 전  
model.add(Dropout(rate = 비율))
'''

# hidden layer1 : [784, 128] -> [input, output]
model.add(Dense(units=128, input_shape = input_shape, activation = 'relu')) # 1층 
model.add(Dropout(rate = 0.3)) # 30% 삭제 

# hidden layer2 : [128, 64] -> [input, output]
model.add(Dense(units=64, activation = 'relu')) # 2층 
model.add(Dropout(rate = 0.1)) # 10% 삭제

# hidden layer3 : [64, 32] -> [input, output]
model.add(Dense(units=32, activation = 'relu')) # 3층
model.add(Dropout(rate = 0.1)) # 10% 삭제

# output layer : [32, 10] -> [input, output]
model.add(Dense(units=10, activation = 'softmax')) # 4층 



# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(60,000) vs val(10,000)
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=15, # 반복학습 
          batch_size = 100, # mini batch
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 



# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)
# 적용 전 : 1s 1ms/step - loss: 0.0934 - accuracy: 0.9795
# 적용 후 : 1s 1ms/step - loss: 0.0754 - accuracy: 0.9792

# 7. model history 
print(model_fit.history.keys())


# loss vs val_loss 
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy 
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()


####################################################################################################

"""
1. Dropout : 무작위 네트워크 삭제 
2. EarlyStopping : loss value에 변화가 없는 경우 학습 조기종료
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense, Dropout # DNN layer 구축 
from tensorflow.keras.callbacks import EarlyStopping # 학습 조기종료

import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() # (images, labels)

x_train.shape # (60000, 28, 28) 
y_train.shape # (60000,)


# 2. x,y변수 전처리 

# 1) x변수 : 정규화 & reshape(3d -> 2d)
x_train = x_train / 255.
x_val = x_val / 255.



# 3d -> 2d : [수정]
x_train = x_train.reshape(-1, 784)
# 28 * 28 = 784
x_train.shape # (60000, 784)

x_val = x_val.reshape(-1, 784)
x_val.shape # (10000, 784)


# 2) y변수 : one hot encoding 
y_train = to_categorical(y_train) 
y_val = to_categorical(y_val) 


# 3. keras model & layer 구축
model = Sequential()


input_shape = (784,) 

# hidden layer1 : [784, 128] -> [input, output]
model.add(Dense(units=128, input_shape = input_shape, activation='relu')) # 1층 
model.add(Dropout(rate = 0.3)) # 30% 제거 

# hidden layer2 : [128, 64] -> [input, output]
model.add(Dense(units=64, activation = 'relu')) # 2층 
model.add(Dropout(rate = 0.1)) # 10% 제거

# hidden layer3 : [64, 32] -> [input, output]
model.add(Dense(units=32, activation = 'relu')) # 3층

# output layer : [32, 10] -> [input, output]
model.add(Dense(units=10, activation = 'softmax')) # 4층 



# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


'''
EarlyStopping 적용 전  
'''

ES = EarlyStopping(monitor='val_loss', patience=2) # 조기종료 환경 

# 5. model training : train(60,000) vs val(10,000) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=30, # 최대 반복학습
          batch_size = 100, # mini batch
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val),
          callbacks = [ES]) # callbacks = [ES] : 조기종료 적용 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 7. model history 
print(model_fit.history.keys())


# loss vs val_loss : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy : : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

####################################################################################################

"""
 Tensorboard : loss value, accuracy 시각화 
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense, Dropout # DNN layer 구축 

import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() # (images, labels)

x_train.shape # (60000, 28, 28) - 3d(size, h, w) 
y_train.shape # (60000,)

# 2. x,y변수 전처리 

# 1) x변수 : 정규화 & reshape(3d -> 2d)
x_train = x_train / 255.
x_val = x_val / 255.


# 3d -> 2d : 모양 변경 
x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)


# 2) y변수 : one hot encoding 
y_train = to_categorical(y_train) 
y_val = to_categorical(y_val) 



# 3. keras model & layer 구축
model = Sequential()


input_dim = (784,) 

# hidden layer1 : [784, 128] -> [input, output]
model.add(Dense(units=128, input_shape = input_dim, activation = 'relu')) # 1층 
model.add(Dropout(rate = 0.3)) 

# hidden layer2 : [128, 64] -> [input, output]
model.add(Dense(units=64, activation = 'relu')) # 2층 
model.add(Dropout(rate = 0.1)) 

# hidden layer3 : [64, 32] -> [input, output]
model.add(Dense(units=32, activation = 'relu')) # 3층
model.add(Dropout(rate = 0.1)) 

# output layer : [32, 10] -> [input, output]
model.add(Dense(units=10, activation = 'softmax')) # 4층 

# model layer 확인 
model.summary()


# 5. model compile : 학습과정 설정(다항분류기) - [수정]
from tensorflow.keras import optimizers

# optimizer='adam' -> default=0.001
model.compile(optimizer=optimizers.Adam(), # default=0.001
              loss = 'categorical_crossentropy', # y : one hot encoding 
              metrics=['accuracy'])


# [추가] 텐서보드(TensorBoard) 
from tensorflow.keras.callbacks import TensorBoard 
from datetime import datetime # 20240624-161023 

# 6. TensorBoard 시각화 
logdir ='c:\\graph\\' + datetime.now().strftime("%Y%m%d-%H%M%S") # 로그파일 경로 
callback = TensorBoard(log_dir=logdir)

model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=20, # 반복학습 
          batch_size = 100, #  mini batch
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val),
          callbacks = [callback]) # 검증셋 


# 7. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 8. model history 
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


# loss vs val_loss : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

