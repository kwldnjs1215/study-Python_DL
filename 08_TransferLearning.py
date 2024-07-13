
"""
딥러닝 이미지넷 분류기 
 - ImageNet으로 학습된 이미지 분류기
"""

# 1. VGGNet(VGG16/VGG19) model 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19 

# 1) model load 
vgg16_model = VGG16(weights='imagenet') 
vgg19_model = VGG19(weights='imagenet') 

# 2) model layer 
vgg16_model.summary()


# 3) model test : 실제 image 적용 
from tensorflow.keras.preprocessing import image # image read 
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# 이미지 로드 
path = r'C:\ITWILL\7_Tensorflow\data\images'
img = image.load_img(path + '/umbrella.jpg', target_size=(224, 224, 3))

X = image.img_to_array(img) # image 데이터 생성 
X = X.reshape(1, 224, 224, 3) # 모양 변경 

# image 전처리 
X = preprocess_input(X)

# image 예측치 
pred = vgg16_model.predict(X)
pred.shape # (1, 1000)

print('predicted :', decode_predictions(pred, top=3))


# 2. ResNet50 model 
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# 1) model load 
resnet50_model = ResNet50(weights='imagenet') 

# 2) model layer 
resnet50_model.summary()


# 이미지 로드 
img = image.load_img(path + '/umbrella.jpg', target_size=(224, 224, 3))

X = image.img_to_array(img) # image 데이터 생성 
X = X.reshape(1, 224, 224, 3)

# image 전처리 
X = preprocess_input(X)

# image 예측치 
pred = resnet50_model.predict(X)
pred.shape # (1, 1000)


print('predicted :', decode_predictions(pred, top=3))


# 3. Inception_v3 model
from tensorflow.keras.applications.inception_v3 import InceptionV3 
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions


# 1) model load 
inception_v3_model = InceptionV3(weights='imagenet') 

# 2) model layer 
inception_v3_model.summary()


# 이미지 로드 
img = image.load_img(path + '/Tank.jpeg', target_size=(299, 299, 3))

X = image.img_to_array(img) # image 데이터 생성 
X = X.reshape(1, 299, 299, 3)

# image 전처리 
X = preprocess_input(X)

# image 예측치 
pred = inception_v3_model.predict(X)
pred.shape # (1, 1000)


print('predicted :', decode_predictions(pred, top=3))


##################################################################################################

"""
전이학습(transfer_learning) : ppt.13 참고 
 - ImageNet 분류기 -> 고양이와 강아지 분류(chap06 > lecture02 > step02) 
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model 

### 단계1 : Feature 추출 
#  - 사전에 학습된 ImageNet 분류 모델에서 가중치 추출 

# 1. VGG16 기본 model 생성 
 
# 1) input image
input_shape = (150, 150, 3) 

# 2) 기본 모델을 이용하여 객체 생성     
base_model = VGG16(weights='imagenet', # 기존 학습된 가중치 이용 
                   input_shape=input_shape,  # input image 
                   include_top = False) # 최상위 레이어(output) 사용안함

# CNN layer 확인 
base_model.summary()

# 3) new model 생성 : 필요한 레이어 선택 
inputs = base_model.layers[0].input # 첫번째 레이어 입력정보  
outputs = base_model.layers[-1].output # 마지막 레이어 출력정보

new_model = Model(inputs, outputs)
 

# 4) new model 가중치 학습여부 지정 
new_model.trainable = False # 모든 레이어의 가중치 학습 동결  


### 단계2 : 전이학습 모델 생성 
# - 학습된 모델의 가중치 이용 -> 입력 이미지 분류 

from tensorflow.keras import Sequential # keras model 
#from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution
from tensorflow.keras.layers import Dense, Flatten # Dropout,  layer
import os


# 1. CNN Model layer 
print('model create')
model = Sequential()


# 2. 전이학습 : 학습된 모델 적용  
model.add(new_model)


''' [기존 CNN 레이터 제외] 
# Convolution layer1 : kernel[3,3,3,32]
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 : kernel[3,3,32,64]
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer3 : kernel[5,5,64,128], maxpooling() 제외 
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten layer :4d -> 2d
model.add(Flatten()) 
# 드롭아웃 - 과적합 해결 
model.add(Dropout(0.5)) # fully connected 층 이전에 배치 
'''

# Flatten layer :3d -> 1d
model.add(Flatten()) 

# Affine layer(Fully connected layer1) : [n, 256]
model.add(Dense(256, activation = 'relu'))

# Output layer(Fully connected layer2) : [256, 1]
model.add(Dense(1, activation = 'sigmoid'))

# model training set : Adam or RMSprop 
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', # one hot encoding
              metrics = ['accuracy'])

# 2. image file preprocessing : 이미지 제너레이터 이용  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("image preprocessing")

# dir setting
base_dir = r"C:\ITWILL\7_Tensorflow\data\images\cats_and_dogs"

train_dir = os.path.join(base_dir, 'train_dir')
validation_dir = os.path.join(base_dir, 'validation_dir')


# image 증식 - 과적합 해결
train_data = ImageDataGenerator(
        rescale=1./255, # 정규화 
        rotation_range = 40, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.2, # image 수평 이동 범위
        height_shift_range = 0.2, # image 수직 이용 범위  
        shear_range = 0.2, # image 전단 각도 범위
        zoom_range=0.2, # image 확대 범위
        horizontal_flip=True,) # image 수평 뒤집기 범위 

# 검증 데이터에는 증식 적용 안함 
validation_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=35, #[수정] batch size 올림
        class_mode='binary') # binary label
# Found 2000 images belonging to 2 classes.

validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=35, # [수정] batch size 올림 
        class_mode='binary')
# Found 1000 images belonging to 2 classes.

# 3. model training : 배치 제너레이터 이용 모델 훈련 
model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=58, #  58*35 = 1epoch 
          epochs=30, # [수정] 30 epochs()
          validation_data=validation_generator,
          validation_steps=29) #  29*35 = 1epoch

# model evaluation
model.evaluate(validation_generator)

# 4. model history graph
import matplotlib.pyplot as plt
 
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

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

##################################################################################################

"""
전이학습(transfer_learning)  : ppt.14~15 참고 
 - cifar10 이미지 분류(chapter06) vs MobileNet 전이학습 
"""

from tensorflow.keras.datasets.cifar10 import load_data # color image dataset 
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras import Sequential # model 생성 
#from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv layer 
from tensorflow.keras.layers import Dense, Flatten, Dropout # DNN layer 
import matplotlib.pyplot as plt 

from tensorflow.keras.applications.mobilenet import MobileNet 

import numpy as np # ndarray
import tensorflow as tf # seed 값
import random 

tf.random.set_seed(35) # seed값 지정
np.random.seed(35)
random.seed(35)


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
#input_shape = (32, 32, 3) # input images 

# 1) model 생성 
model = Sequential()

# 2) layer 구축 
'''
# Conv layer1 
model.add(Conv2D(filters=32, kernel_size=(5,5), 
                 input_shape = input_shape, activation='relu')) # image[28x28]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[13x13]

# Conv layer2 
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # image[9x9]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[4x4]

# Conv layer3  
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))# image[2x2]
'''


###########################################
####  전이학습 
###########################################

### MobileNet 기본 model 생성 

# 1) input image
input_shape = (32, 32, 3) 

# 2) 기본 모델을 이용하여 객체 생성 : 학습된 가중치, 입력 이미지 정보, 출력층(x)    

base_model = MobileNet(weights='imagenet', # 기존 학습된 가중치 이용 
                   input_shape=input_shape,  # input image 
                   include_top = False) # 최상위 레이어(output) 사용안함

model.add(base_model) # 전이학습 모델 레이어 추가 


# 전결합층 : Flatten layer 
model.add(Flatten())

model.add(Dropout(0.7))

# DNN1 : hidden layer 
model.add(Dense(units=64, activation='relu'))


# DNN2 : output layer  
model.add(Dense(units = 10, activation='softmax'))
                  

# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(50000) vs val(10000) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=10, # 반복학습 
          batch_size = 100, # 1회 공급 image size 
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

##################################################################################################

"""
  <transformers 설치 과정> 
1. conda activate tensorflow # 가상환경 활성화 
2. pip install transformers # transformers 설치 
"""

from transformers import pipeline # data input > token -> model 
'''
 관련 사이트 : https://huggingface.co/t5-base
 학습된 모델을 이용하므로 메모리를 많이 차지함(Colab 사용 권장)  
'''

### 1. 감성분석/문서분류 : 비지도학습 
'''
감성분석 : 주어진 문장(문서)의 성질/성향 분석(label : 긍정,중립,부정)
문서분류 : 주어진 문장(문서)를 category로 분류 
'''
# 1) 감성분석 
sentiment = pipeline(task = "sentiment-analysis")

pred = sentiment("what a beautiful day!") 
pred = pred[0] # dict 반환 

print(f'감성분석 결과 : {pred["label"]}, 감성 점수 : {pred["score"]}')

# 2) 문서분류 
classifier = pipeline(task = "text-classification") 
texts = ["This restaurant is awesome", "This restaurant is awful"]

preds = classifier(texts) # args (`str` or `List[str]`)
preds

for pred, text in zip(preds, texts) :
    print(f'{text} -> 분류결과 : {pred["label"]}, 감성 점수 : {pred["score"]}')



### 2. 텍스트 문서생성 
text_generator = pipeline(task = "text-generation") # or model="gpt2"
   
text_inputs =  """The COVID-19 pandemic, The novel virus was first identified in an outbreak in the Chinese city of Wuhan in 
December 2019."""

pred_text = text_generator(text_inputs)
print(pred_text) # list of `dict`

pred_text = pred_text[0]['generated_text'] # value 반환 
generated_text = pred_text.removeprefix(text_inputs) # input text 제거 

'''
import sys 
import time
for i in range(len(generated_text)):      
    next_char = generated_text[i] # 다음 글자 예측 
    time.sleep(0.1) # 0.1초 interval
    sys.stdout.write(next_char) # 줄바꿈없이 바로 이어서 출력
    sys.stdout.flush()
'''    
    
### 3. 질의응답(question-answering) 
question_answerer = pipeline(task = "question-answering")
'''
추출형 질의응답(extractive question answering)
문서에 대해서 질문을 제시하고 문서 자체에 존재하는 텍스트 범위(spans of text)를 
해당 질문에 대한 답변으로 추출하는 작업
'''
   
context = """Text mining, also referred to as text data mining, similar to text analytics,
is the process of deriving high-quality information from text. It involves
the discovery by computer of new, previously unknown information,
by automatically extracting information form different written resources."""


question = input('input question : ')
answer = question_answerer(question=question, context=context)
print(answer['answer'])


### 4. 문서 요약(summarization)
summarizer = pipeline(task = "summarization")

texts ="""Deep Neural Networks, also called convolutional networks, are composed of multiple levels of nonlinear operations, such as neural nets with many hidden layers. Deep learning methods aim at learning feature hierarchies, where features at higher levels of the hierarchy are formed using the features at lower levels. In 2006, Hinton et al. proved that much better results could be achieved in deeper architectures when each layer is pretrained with an unsupervised learning algorithm. Then the network is trained in a supervised mode using back-propagation algorithm to adjust weights. Current studies show that DNNs outperforms GMM and HMM on a variety of speech processing tasks by a large margin"""

result_texts = summarizer(texts)  # args (`str` or `List[str]`):           

summary_text = result_texts[0]['summary_text']
print(summary_text)

    
### 5. 문서번역(translation) 
'''
 1) 영어 -> 프랑스어 번역 
'''
translator_fr = pipeline(task = "translation_en_to_fr")
        
result_texts = translator_fr("How old are you?")
translation_text = result_texts[0]['translation_text']
print(translation_text)

'''
 2) 영어 -> 독일어 번역 
'''
translator_de = pipeline(task = "translation_en_to_de")

result_texts = translator_de("How old are you?")
translation_text = result_texts[0]['translation_text']
print(translation_text)

# 긴 문장 텍스트 
long_text = "The process of handling text data is a little different compared to other problems. This is because the data is usually in text form. You ,therefore, have to figure out how to represent the data in a numeric form that can be understood by a machine learning model. In this article, let's take a look at how you can do that. Finally, you will build a deep learning model using TensorFlow to classify the text."
translator_de(long_text)


### 6. 개체명 인식(Named-entity recognition)
'''
 - 문장에서 여러 개체(entity)를 감지하여 인식된 각 엔터티의 끝점과 신뢰도 점수 반환
'''
ner = pipeline("ner")

text = "John works for Google that is located in the USA"
result_texts = ner(text)
result_texts

