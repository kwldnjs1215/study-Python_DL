
"""
RNN model 
 - 순환신경망 Many to One RNN 모델(PPT.8 참고)  
"""

import tensorflow as tf # seed value 
import numpy as np # ndarray
from tensorflow.keras import Sequential # model
from tensorflow.keras.layers import SimpleRNN, Dense # RNN layer 

import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(34)
np.random.seed(34)
rd.seed(34)


# many-to-one : word(4개) -> 출력(1개)
X = [[[0.0], [0.1], [0.2], [0.3]], 
     [[0.1], [0.2], [0.3], [0.4]],
     [[0.2], [0.3], [0.4], [0.5]],
     [[0.3], [0.4], [0.5], [0.6]],
     [[0.4], [0.5], [0.6], [0.7]],
     [[0.5], [0.6], [0.7], [0.8]]] 

y = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X.shape # (6, 4, 1) : RNN 3차원 입력(batch_size, time_steps, features)

model = Sequential() 

input_shape = (4, 1) # (timestep, feature)

# RNN layer 추가 
model.add(SimpleRNN(units=35, input_shape=input_shape, 
                    return_state=False, # Many to One
                    activation='tanh'))

# DNN layer 추가 
model.add(Dense(units=1)) # 출력 : 회귀모델 

# model 학습환경 
model.compile(optimizer='adam', 
              loss='mse', metrics=['mae'])

# model training 
model.fit(X, y, epochs=50, verbose=1)

# model prediction
y_pred = model.predict(X)
print(y_pred)


############################################################################################################

"""
 - 시계열분석 : 시계열데이터 + RNN model (PPT.14~15 참고)
"""
import pandas as pd # csv file read 
import matplotlib.pyplot as plt # 시계열 시각화 
import numpy as np # ndarray
import tensorflow as tf # seed 값 
from tensorflow.keras import Sequential # model 
from tensorflow.keras.layers import SimpleRNN, Dense # RNN layer 
import random 

tf.random.set_seed(35) # seed값 지정
np.random.seed(35)
random.seed(35)
 

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. csv file read  
path = r'C:\ITWILL\7_Tensorflow\data'
timeSeries = pd.read_csv(path + '/timeSeries.csv')

data = timeSeries['data']
print(data) 


# 2. RNN 적합한 dataset 생성 : : (batch_size, time_steps, features)
x_data = [] # 학습데이터 : 1~10개 시점 데이터  
for i in range(len(data)-10) : # 90
    for j in range(10) : # 10
        x_data.append(data[i+j]) # 90 * 10 = 900

# list -> array 
x_data = np.array(x_data)


y_data = [] # 정답 : 11번째 시점 데이터  
for i in range(len(data)-10) : # 90
    y_data.append(data[i+10]) # 90


# list -> array
y_data = np.array(y_data)


# train(70)/val(20) split 
x_train = x_data[:700].reshape(70, 10, 1) 
x_val = x_data[700:].reshape(20, 10, 1) 


# train(70)/val(20) split 
y_train = y_data[:70].reshape(70,1) 
y_val = y_data[70:].reshape(20,1) 



# 3. model 생성 
model = Sequential()

input_shape = (10, 1)

# RNN layer 추가 
model.add(SimpleRNN(units=16, input_shape=input_shape, 
                    activation ='tanh'))

# DNN layer 추가 
model.add(Dense(units=1)) # # 회귀모델 : 활성함수 없음 

# model 학습환경 
model.compile(optimizer='sgd', 
              loss='mse', metrics=['mae'])

# model 학습 
model.fit(x=x_train, y=y_train, 
          epochs=400, verbose=1)


# model 예측 
y_pred = model.predict(x_val) 


# 학습데이터(70개) + 예측데이터(20개) 시각화 
y_pred = np.concatenate([y_train, y_pred])  


# 20개 주가 예측 
future = 20

threshold = np.ones_like(y_pred, dtype='bool') 
threshold[:-future] = False  

pred_x = np.arange(len(y_pred)).reshape(-1, 1) # x축 색인자료 
pred_y = y_pred # y축 시계열 예측치  

# 한글 & 음수부호 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# y_true vs y_pred 
plt.plot(y_data, color='lightblue', linestyle='--', marker='o', label='real value')
plt.plot(pred_x[threshold], pred_y[threshold], 'r--', marker='o', label='predicted value')
plt.legend(loc='best')
plt.title(f'{20}개의 시계열 예측결과')
plt.show()

############################################################################################################

"""
기후 데이터 시계열 분석 : ppt.17 참고 

시계열 모델의 독립변수와 종속변수 
 독립변수 : 이전 시점 20개 온도 -> 종속변수 : 21번째 온도(1개)  
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# Matplotlib Setting
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
tf.random.set_seed(13) # Random Seed Setting


'''
에나 기후(jena climate) dataset
독일 에나 연구소에서 제공하는 기후(climate) 데이터셋으로 온도, 기압, 습도 등 14개의 
날씨 관련 변수를 제공한다. 8년(2009~2016) 동안 매일 10 단위로 기록한 데이터셋 
'''

# 1. csv file read : 에나 기후(jena climate) dataset 
path = r'C:\ITWILL\7_Tensorflow\data'
df = pd.read_csv(path+'/jena_climate_2009_2016.csv')
df.info() # 420551
'''
RangeIndex: 420551 entries, 0 to 420550
Data columns (total 15 columns):
 #   Column           Non-Null Count   Dtype  
---  ------           --------------   -----  
 0   Date Time        420551 non-null  object  : 날짜/시간 
 1   p (mbar)         420551 non-null  float64 : 대기압(밀리바 단위)
 2   T (degC)         420551 non-null  float64 : 온도(섭씨)
 3   Tpot (K)         420551 non-null  float64 : 온도(절대온도)
 4   Tdew (degC)      420551 non-null  float64 : 습도에 대한 온도
 5   rh (%)           420551 non-null  float64 : 상대 습도
 6   VPmax (mbar)     420551 non-null  float64 : 포화증기압
 7   VPact (mbar)     420551 non-null  float64 : 중기압 
 8   VPdef (mbar)     420551 non-null  float64 : 중기압부족 
 9   sh (g/kg)        420551 non-null  float64 : 습도 
 10  H2OC (mmol/mol)  420551 non-null  float64 : 수증기 농도 
 11  rho (g/m**3)     420551 non-null  float64 : 공기밀도 
 12  wv (m/s)         420551 non-null  float64 : 풍속 
 13  max. wv (m/s)    420551 non-null  float64 : 최대풍속
 14  wd (deg)         420551 non-null  float64 : 풍향 
''' 


#######################################
# LSTM을 이용한 기상예측: 단변량
#######################################

### 1. 변수 선택 및 탐색  
uni_data = df['T (degC)'] # 온도 칼럼 
uni_data.index = df['Date Time'] # 날짜 칼럼으로 index 지정 


# Visualization the univariate : 표준화 필요성 확인 
uni_data.plot(subplots=True)
plt.show() # -20 ~ 40

# 시계열 자료 추출 
uni_data = uni_data.values # 값 추출 

# 표준화(Z-Normalization)   
uni_train_mean = uni_data.mean()
uni_train_std = uni_data.std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

# 표준화 여부 확인 
plt.plot(uni_data)
plt.show() # -3 ~ 3



### 2. 단변량 데이터 생성 :  LSTM모델 공급에 적합한 자료 만들기 

# 1) 단변량 데이터 생성 함수 
def univariate_data(dataset, s_index, e_index, past_size) : 
    X = [] # x변수 
    y = [] # y변수 

    s_index = s_index + past_size
    if e_index is None: # val dataset 
        e_index = len(dataset) 
    
    for i in range(s_index, e_index): 
        indices = range(i-past_size, i) 
        X.append(np.reshape(dataset[indices], (past_size, 1))) # x변수(20, 1)  
        
        y.append(dataset[i]) # y변수(1,)  
        
    return np.array(X), np.array(y)


# 2) 단변량 데이터 생성 
TRAIN_SPLIT = 300000 # train vs val split 기준
past_data = 20 # x변수 : 과거 20개 자료[0~19, 1~20,..] 

# 훈련셋 
X_train, y_train = univariate_data(uni_data,0,TRAIN_SPLIT,past_data)
# 검증셋 
X_val, y_val = univariate_data(uni_data,TRAIN_SPLIT, None, past_data)

# Check the Data
print(X_train.shape) # (299980, 20, 1) 
print(y_train.shape) # (299980,) 



### 3. LSTM Model 학습 & 평가   
input_shape=(20, 1)

model = Sequential()
model.add(LSTM(16, input_shape = input_shape)) 
model.add(Dense(1)) # 회귀함수
model.summary()


# 학습환경 
model.compile(optimizer='adam', loss='mse')


# 모델 학습 
model_history = model.fit(X_train, y_train, epochs=10, # trainset
          batch_size = 256,
          validation_data=(X_val, y_val))#, # valset 

          
# model evaluation 
print('='*30)
print('model evaluation')
model.evaluate(x=X_val, y=y_val)



### 4. Model 손실(Loss) 시각화 
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Single Step Training and validation loss')
plt.legend(loc='best')
plt.show()


### 5. model prediction 

# 테스트셋 선택 : 검증셋 중에서 5개 관측치 선택 
X_test = X_val[:5] # (5, 20, 1)
y_test = y_val[:5] # (5,)


# 예측치 : 5개 관측치 -> 5개 예측치 
y_pred = model.predict(X_test)  

############################################################################################################

"""

스팸 메시지 분류기 : ppt.33 참고 
"""

# texts 처리 
import pandas as pd # csv file
import numpy as np # list -> numpy 
import string # texts 전처리  
from sklearn.model_selection import train_test_split # split
from tensorflow.keras.preprocessing.text import Tokenizer # 토큰 생성기 
from tensorflow.keras.preprocessing.sequence import pad_sequences # 패딩 

# DNN model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM # 순환신경망 

# 1. csv file laod 
path = r'C:\ITWILL\7_Tensorflow\data'
spam_data = pd.read_csv(path + '/spam_data.csv', header = None)


label = spam_data[0] 
texts = spam_data[1]


# 2. texts와 label 전처리

# 1) label 전처리 
label = [1 if lab=='spam' else 0  for lab in label]

# list -> numpy 형변환 
label = np.array(label)

# 2) texts 전처리 
def text_prepro(texts): # [text_sample.txt] 참고 
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in string.digits) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return texts


# 함수 호출 
texts = text_prepro(texts)
print(texts)


# 3. 단어 생성기 
tokenizer = Tokenizer()  

tokenizer.fit_on_texts(texts = texts) # 텍스트 반영 -> token 생성  

words = tokenizer.index_word # 단어 반환 
print(words)

words_size = len(words) + 1 # 전체 단어수+1 



# 4. 정수색인 : 단어 고유숫자 변환 
seq_result = tokenizer.texts_to_sequences(texts)
print(seq_result)

# 최대 단어길이 
lens = [len(sent) for sent in seq_result]
print(lens)

maxlen = max(lens)


# 5. padding : maxlen 기준으로 모든 문장의 단어 길이 맞춤 
x_data = pad_sequences(seq_result, maxlen = maxlen)
x_data.shape # (5574, 171) 


# 6. train/test split : 80% vs 20%
x_train, x_val, y_train, y_val = train_test_split(
    x_data, label, test_size=20)


# 임베딩 차원 : 16, 32, 64, 128,…1024차원(단어가 많은 경우)
embedding_dim = 32 


# 7. DNN model & layer
model = Sequential()  

# Embedding layer : 1층 
model.add(Embedding(input_dim=words_size, 
                    output_dim=embedding_dim, 
                    input_length=maxlen))


# 순환신경망(RNN layer) 
model.add(LSTM(units= 64, activation='tanh')) # 2층 


# hidden layer1 : w[64, 32] 
model.add(Dense(units=32,  activation='relu')) # 3층 

# output layer : [32, 1]
model.add(Dense(units = 1, activation='sigmoid')) # 4층 
          

# 8. model compile : 학습과정 설정(이항분류기)
model.compile(optimizer='adam', 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])

# 9. model training : train(80) vs val(20) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=5, # 반복학습 
          batch_size = 512,
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 10. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

