
"""
합성곱과 폴링 : ppt. 21~22 참고

합성곱층(Convolution layer) = 합성곱 + 폴링   
 합성곱(Convolution) : 이미지 특징 추출
 폴링(Pooling) : 이미지 픽셀 축소(다운 샘플링)
"""
import tensorflow as tf # 합성곱, 폴링 연산 
from tensorflow.keras.datasets.mnist import load_data # 데이터셋 
import numpy as np # 이미지 축 변경 
import matplotlib.pyplot as plt # 이미지 시각화 

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Input image 만들기  
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 


# 1) 자료형 변환 : int -> float
x_train = x_train.astype('float32') # type 일치 
x_test = x_test.astype('float32')

# 2) 정규화
x_train.min() # 0.0 -> 흰색 
x_train.max() # 255.0 -> 검정색 
x_train /= 255 # x_train = x_train / 255
x_test /= 255


# 3) input image 선정 : 첫번째 image 
img = x_train[0]
plt.imshow(img, cmap='gray') # 숫자 5  
plt.show() 
img.shape # (28, 28)

# 4) input image 모양변경  
inputImg = img.reshape(1,28,28,1) # [size, h, w, c]


# 2. Filter 만들기 : image에서 특징 추출  
Filter = tf.Variable(tf.random.normal([3,3,1,5])) # [h, w, c, fmap] 
'''
h=3, w=3 : 커널(kernel) 세로,가로 크기
c=1 : 이미지 채널(channel)수    
fmap=5 : 추출할 특징 맵 개수   
'''


# 3. Convolution layer : 특징맵  추출     
conv2d = tf.nn.conv2d(inputImg, Filter, strides=[1,1,1,1],
                      padding='SAME') # 이미지와 필터 합성곱
'''
strides=[1,1,1,1] : kernel 가로/세로 1칸씩 이동 
padding = 'SAME' : 원본이미지와 동일한 크기로 이미지 특징 추출 
'''

conv2d.shape # [1, 28, 28, 5] # padding='SAME'
conv2d.shape # [1, 26, 26, 5] # padding='VALID'
'''
output = (Input_size - Kernel_size) / Stride + 1
'''
output = (28 - 3) / 1 + 1
output # 26.0

# 합성곱(Convolution) 연산 결과  
conv2d_img = np.swapaxes(conv2d, 0, 3) # 축 변환(첫번째와 네번째) 

for i, img in enumerate(conv2d_img) : 
    plt.subplot(1, 5, i+1) # 1행 5열, 열index 
    plt.imshow(img, cmap='gray')  
plt.show()



# 4. Pool layer : 특징맵 픽셀 축소 
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1],
                      padding='SAME')
'''
ksize=[1,2,2,1] : pooling 크기 가로/세로 2칸씩 이동(이미지 1/2 축소) 
padding = 'SAME' : 원본이미지와 동일한 크기로 특징 이미지 추출 
'''  
'''
Output = (Input_size – Pool_size) / Stride + 1
'''

pool.shape # [1, 14, 14, 5]
Output = (28 - 2) / 2 + 1 
Output # 14.0

# 폴링(Pool) 연산 결과 
pool_img = np.swapaxes(pool, 0, 3) # 축 변환(첫번째와 네번째)

for i, img in enumerate(pool_img) :
    plt.subplot(1,5, i+1)
    plt.imshow(img, cmap='gray') 
plt.show()

    
################################################################################################################

"""
real image + CNN basic
1. Convolution layer : image 특징 추출  
  -> Filter : 9x9 
  -> 특징맵 : 5개 
  -> strides = 2x2, padding='SAME'
2. Pooling layer : image 축소 
  -> ksize : 7x7
  -> strides : 4x4
  -> padding='SAME' 
"""

import tensorflow as tf # ver2.x

import numpy as np
import matplotlib.pyplot as plt # image print
from matplotlib.image import imread # image read

# 1. image load 
img = imread("C:/ITWILL/7_Tensorflow/data/images/parrots.png")
plt.imshow(img)
plt.show()

# 2. image shape & RGB 픽셀값 
print(img.shape) # (512, 768, 3) 
print(img)
 

# 3. input image 만들기   
Img = img.reshape(1, 512, 768, 3) # (size, h, w, c)



# 4. Filter 만들기  
Filter = tf.Variable(tf.random.normal([9,9,3,5])) # [h, w, c, fmap]


# 5. Convolution layer 
conv2d = tf.nn.conv2d(Img, Filter, 
                      strides=[1,2,2,1], padding='SAME')

conv2d.shape # [1, 256, 384, 5]
'''
padding = 'VALID' 일때 : ppt.12 참고 
output = (Input_size - Kernel_size) / Stride + 1

padding='SAME' 일때 : ppt.13 참고 
output = (Input_size) / Stride 
'''
# 특징맵 가로/세로 픽셀 구하기 
h_output = 512 / 2 # 256.0
w_output = 768 / 2 # 384.0


# 합성곱(Convolution) 연산 결과 
conv2d_img = np.swapaxes(conv2d, 0, 3)
conv2d_img.shape # (5, 256, 384, 1)  

fig = plt.figure(figsize = (20, 6))  
for i, img in enumerate(conv2d_img) :
    fig.add_subplot(1, 5, i+1) 
    plt.imshow(img) 
plt.show()


# 6. Pool layer 
pool = tf.nn.max_pool(conv2d, ksize=[1,7,7,1], 
                      strides=[1,4,4,1], padding='SAME')

 
pool.shape # [1, 64, 96, 5]

'''
padding = 'VALID' 일때 : ppt.17 참고
output = (Input_size – Pool_size) / Stride + 1

padding='SAME' 일때 
output = (Input_size) / Stride
'''

# 폴링 세로/가로 픽셀 구하기 
h_output = (256) / 4  # 64
w_output = (384) / 4  # 96


# 폴링(Pool) 연산 결과 
pool_img = np.swapaxes(pool, 0, 3)

fig = plt.figure(figsize = (20, 6))    
for i, img in enumerate(pool_img) :
    fig.add_subplot(1,5, i+1)
    plt.imshow(img) 
plt.show()
    
################################################################################################################

"""
CNN model 생성 
 1. image dataset load 
 2. image dataset 전처리 
 3. CNN model 생성 : layer 구축 + 학습환경 + 학습 
 4. CNN model 평가
 5. CMM model history 
"""

from tensorflow.keras.datasets.cifar10 import load_data # color image dataset 
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv layer 
from tensorflow.keras.layers import Dense, Flatten # DNN layer 
import matplotlib.pyplot as plt 

# 1. image dataset load 
(x_train, y_train), (x_val, y_val) = load_data()

x_train.shape # image : (50000, 32, 32, 3) - (size, h, w, c)
y_train.shape # label : (50000, 1)


x_val.shape # image : (10000, 32, 32, 3)
y_val.shape # label : (10000, 1)


# 2. image dataset 전처리

# 1) image pixel 실수형 변환 
x_train = x_train.astype(dtype ='float32')  
x_val = x_val.astype(dtype ='float32')

# 2) image 정규화 : 0~1
x_train = x_train / 255
x_val = x_val / 255


# 3) label 전처리 : 10진수 -> one hot encoding(2진수) 
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)


# 3. CNN model & layer 구축
x_train.shape # (50000, 32, 32, 3)
 
input_shape = (32, 32, 3) # input images 

# 1) model 생성 
model = Sequential()

# 2) layer 구축 
# Conv layer1 : Conv + MaxPool
model.add(Conv2D(filters=32, kernel_size=(5, 5),                  
                 input_shape = input_shape, activation='relu')) 
'''
filters=32 : 특징맵 32장 
kernel_size=(5, 5) : 커널 가로/세로 크기 
strides = (1, 1) (기본값) : 커널 가로/세로 이동 크기
padding = 'valid'(기본값) : 합성곱으로 특징맵 크기 결정   
'''
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2))) 
'''
pool_size = (3, 3) : pooling window 크기 
strides=(2, 2) : pooling window 가로/세로 이동 크기
padding = 'valid'(기본값) : stride에 의해서 특징맵 크기 결정 
'''

# Conv layer2 : Conv + MaxPool 
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu')) 
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2))) 

# Conv layer3 : Conv   
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

# 전결합층 : Flatten layer 
model.add(Flatten()) # 3d/2d -> 1d  

# DNN1 : hidden layer 
model.add(Dense(units=64, activation='relu')) # 4층 

# DNN2 : output layer  
model.add(Dense(units = 10, activation='softmax')) # 5층 
                  

model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
              input image    (100, 32, 32, 3)       
 conv2d_3 (Conv2D)           (None, 28, 28, 32)        2432 
     
 output = (Input_size - Kernel_size) / Stride + 1
 output = (32 - 5) / 1 + 1 # 28.0
                                                              
 max_pooling2d_2 (MaxPooling  (None, 13, 13, 32)       0         
 2D)         
                                                    
 output = (Input_size – Pool_size) / Stride + 1
 output = (28 - 3) / 2 + 1 # 13
                                                                
 conv2d_4 (Conv2D)           (None, 9, 9, 64)          51264     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 4, 4, 64)         0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 2, 2, 128)         73856     
                                                                 
 flatten_1 (Flatten)         (None, 512) = 2*2*128     0         
                                                                 
 dense_2 (Dense)             (None, 64)                32832     
                                                                 
 dense_3 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 161,034
'''



# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(105) vs val(45) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋(50,000) 
          epochs=10, # 반복학습 : 50000*10 = 500000
          batch_size = 100, # 1회 공급 image size 100*500=50000
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 6. CNN model 평가 : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

 
# 7. CMM model history 
print(model_fit.history.keys()) # key 확인 
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


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


################################################################################################################

"""
 - Keras CNN layers tensorboard 시각화 
"""
import tensorflow as tf 
from tensorflow.keras.datasets.cifar10 import load_data # color image dataset 
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv layer 
from tensorflow.keras.layers import Dense, Flatten, Dropout # DNN layer 
import matplotlib.pyplot as plt 

# [추가] tensorboard 초기화 
tf.keras.backend.clear_session()


# 1. image dataset load 
(x_train, y_train), (x_val, y_val) = load_data()

x_train.shape # image : (50000, 32, 32, 3) - (size, h, w, c)
y_train.shape # label : (50000, 1)

first_img = x_train[0]
first_img.shape # (32, 32, 3)

plt.imshow(first_img)
plt.show()

print(y_train[0]) # [6]

x_val.shape # image : (10000, 32, 32, 3)
y_val.shape # label : (10000, 1)


# 2. image dataset 전처리

# 1) image pixel 실수형 변환 
x_train = x_train.astype(dtype ='float32') # type일치  
x_val = x_val.astype(dtype ='float32')

# 2) image 정규화 : 0~1
x_train = x_train / 255
x_val = x_val / 255

x_train[0]

# 3) label 전처리 : 10진수 -> one hot encoding(2진수) 
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

y_train.shape # (50000, 10)
y_train[0] # [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]


# 3. CNN model & layer 구축 
input_shape = (32, 32, 3)

# 1) model 생성 
model = Sequential()

# 2) layer 구축 
# Conv layer1 : filter[5, 5, 3, 32]
model.add(Conv2D(filters=32, kernel_size=(5,5), 
                 input_shape = input_shape, activation='relu')) # image[28x28]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[13x13]
model.add(Dropout(0.3))

# Conv layer2 : filter[5, 5, 32, 64]
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # image[9x9]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[4x4]
model.add(Dropout(0.1))

# Conv layer3 : filter[3, 3, 64, 128] - MaxPool2D 제외 
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))# image[2x2]
model.add(Dropout(0.1))


# 전결합층 : Flatten layer : 3d[h,w,c] -> 1d[n=h*w*c]
model.add(Flatten())


# DNN : hidden layer : 4층[n x 64] 
model.add(Dense(units=64, activation='relu'))


# DNN : output layer : 5층 
model.add(Dense(units = 10, activation='softmax'))
        
model.summary()  
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_9 (Conv2D)            (None, 28, 28, 32)        2432      
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 9, 9, 64)          51264     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
dropout_9 (Dropout)          (None, 4, 4, 64)          0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 2, 2, 128)         73856     
_________________________________________________________________
dropout_10 (Dropout)         (None, 2, 2, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                32832     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
'''
          

# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# [추가] Tensorboard 
from tensorflow.keras.callbacks import TensorBoard 
from datetime import datetime # 날짜/시간 자료 생성 

logdir = 'c:/graph/' + datetime.now().strftime('%Y%m%d-%H%M%S') # '년월일-시분초'
callback = TensorBoard(log_dir=logdir)


# 5. model training : train(105) vs val(45) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=2, # 반복학습 
          batch_size = 100, # 1회 공급 image size
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val), # 검증셋
          callbacks = [callback]) # [추가] 


# 6. CNN model 평가 : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

 
# 7. CMM model history 
print(model_fit.history.keys()) # key 확인 
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


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


'''
Tensorboard 실행순서
1. logdir : log 파일 확인 
2. (base) conda activate tensorflow 
3. (tensorflow) tensorboard --logdir=C:\graph\년월일-시분초\train
-> 주의 : '=' 앞과 뒤 공백 없음 
4. http://localhost/6006
'''

################################################################################################################

"""
Cats vs Dogs image classifier 
 - image data generator 이용 : 학습 데이터셋 만들기 
"""
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution layer
from tensorflow.keras.layers import Dense, Flatten # Affine layer
import os # 경로 지정 


# image resize
img_h = 150 # height
img_w = 150 # width
input_shape = (img_h, img_w, 3) 

# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer3 : maxpooling() 제외 
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten layer : 3d -> 1d
model.add(Flatten()) 

# DNN hidden layer(Fully connected layer)
model.add(Dense(256, activation = 'relu'))

# DNN Output layer
model.add(Dense(1, activation = 'sigmoid'))

# model training set  
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])


# 2. image file preprocessing : image 생성   
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dir setting
base_dir = "C:\\ITWILL\\7_Tensorflow\\data\\images\\cats_and_dogs"

train_dir = os.path.join(base_dir, 'train_dir') # 훈련용 이미지 
validation_dir = os.path.join(base_dir, 'validation_dir') # 검증용 이미지 


# 훈련셋 이미지 생성기 
train_data = ImageDataGenerator(rescale=1./255) # 정규화 

# 검증셋 이미지 생성기
validation_data = ImageDataGenerator(rescale=1./255) # 정규화

train_generator = train_data.flow_from_directory(
        train_dir, # 훈련용 이미지 경로 
        target_size=(150,150), # 이미지 규격화 
        batch_size=20, # 공급용 데이터 크기 
        class_mode='binary') # 이항분류 
# Found 2000 images belonging to 2 classes.

validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')
# Found 1000 images belonging to 2 classes.


# 3. model training 
model_fit = model.fit_generator( # model.fit()    
          train_generator, # 훈련용 이미지 공급 
          steps_per_epoch=100, # batch_size 반복횟수 = 20*100 = 2000 
          epochs=10, # 10 * 2000 = 20000
          validation_data=validation_generator, # 검증용 이미지 공급 
          validation_steps=50) # 20*50 = 1000

# model evaluation
model.evaluate(validation_generator)


# 4. model history graph
import matplotlib.pyplot as plt
 
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(model_fit.history.keys())

loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']


## 3epoch 과적합 시작점 
epochs = range(1, len(acc) + 1) # range(1, 11)

# acc vs val_acc   
plt.plot(epochs, acc, 'b--', label='train acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('Training vs validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='best')
plt.show()

# loss vs val_loss 
plt.plot(epochs, loss, 'b--', label='train loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('Training vs validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

################################################################################################################

"""
model overfitting solution
"""
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution
from tensorflow.keras.layers import Dense, Dropout, Flatten # layer
import os

# Hyper parameters
img_h = 150 # height
img_w = 150 # width
input_shape = (img_h, img_w, 3) 

# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer3  
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten layer :3d -> 1d
model.add(Flatten()) 

# 2차 적용 : 드롭아웃 - 과적합 해결
model.add(Dropout(0.5))


# DNN layer1 : Affine layer(Fully connected layer1) 
model.add(Dense(256, activation = 'relu'))

# DNN layer2 : Output layer(Fully connected layer2)
model.add(Dense(1, activation = 'sigmoid'))

# model training set 
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])


# 2. image file preprocessing : 이미지 제너레이터 이용  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dir setting
base_dir = "C:\\ITWILL\\7_Tensorflow\\data\\images\\cats_and_dogs"

train_dir = os.path.join(base_dir, 'train_dir')
validation_dir = os.path.join(base_dir, 'validation_dir')

# 1차 적용 
#train_data = ImageDataGenerator(rescale=1./255)

# 2차 적용 : image 증식 - 과적합 해결
train_data = ImageDataGenerator(
        rescale=1./255, # 정규화 
        rotation_range = 40, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.2, # image 수평 이동 범위
        height_shift_range = 0.2, # image 수직 이용 범위  
        shear_range = 0.2, # image 전단 각도 범위
        zoom_range=0.2, # image 확대 범위
        horizontal_flip=True) # image 수평 뒤집기 범위 

# 검증 데이터에는 증식 적용 안함 
validation_data = ImageDataGenerator(rescale=1./255)

# 훈련셋 생성 
train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=35, #[수정] 
        class_mode='binary') 

# 검증셋 생성 
validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=35, # [수정] 
        class_mode='binary')


# 3. model training : 배치 제너레이터 이용 모델 훈련 
model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=58, #  58*35 = 1epoch 
          epochs=30, # [수정] 
          validation_data=validation_generator,
          validation_steps=29) #  29*35 = 1epoch

# model evaluation
model.evaluate(validation_generator)

# 4. model history graph
import matplotlib.pyplot as plt
 
loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']

epochs = range(1, len(acc) + 1)

# acc vs val_acc   
plt.plot(epochs, acc, 'b--', label='train acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('Training vs validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='best')
plt.show()

# loss vs val_loss 
plt.plot(epochs, loss, 'b--', label='train loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('Training vs validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

