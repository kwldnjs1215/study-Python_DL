
"""
셀레리움 설치 
(base) conda activate tensorflow 
(tensorflow) pip install selenium
(tensorflow) pip install webdriber_manager
"""

from selenium import webdriver # 드라이버 
from selenium.webdriver.chrome.service import Service # Chrom Service
from webdriver_manager.chrome import ChromeDriverManager # 크롬브라우저 관리자  
import time # 화면 일시 정지

 
# 1. driver 객체 생성
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
# driver 객체 생성 : 크롬브라우저 창 띄움   

dir(driver)
'''
driver.get(url) # url 이동 
driver.close() # 창 닫기 
'''

# 2. 대상 url 이동 
driver.get('https://www.naver.com/') # url 이동 

# 3. 일시 중지 & driver 종료 
time.sleep(5) # 5초 일시 중지 
driver.close() # 현재 창 닫기  

###########################################################################################

"""
1. google 페이지 이동 
2. 입력상자 가져오기 
3. 검색 입력 -> 엔터  
4. 검색 페이지 이동
"""

from selenium import webdriver # driver 
from selenium.webdriver.chrome.service import Service # Chrom 서비스
from webdriver_manager.chrome import ChromeDriverManager # 크롬드라이버 관리자 
from selenium.webdriver.common.by import By # 로케이터(locator) 제공
from selenium.webdriver.common.keys import Keys # 엔터키 사용(Keys.ENTER) 
import time

# 1. driver 객체 생성 
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))


# 2. 대상 url 이동 
driver.get('https://www.google.co.kr/') # google 페이지 이동 


# 3. 검색 입력상자 tag -> 검색어 입력   
search_box = driver.find_element(By.NAME, 'q')
search_box.send_keys('딥러닝') # 검색어 입력     
search_box.send_keys(Keys.ENTER)

# 4. 검색 결과 페이지 
time.sleep(3) # 3초 대기(자원 loading)

driver.close() # 현재 창 닫기  

###########################################################################################

"""
셀럽 이미지 수집 
 Selenium + Driver + BeautifulSoup
"""

from selenium import webdriver # driver 
from selenium.webdriver.chrome.service import Service # Chrom 서비스
from webdriver_manager.chrome import ChromeDriverManager # 크롬드라이버 관리자
from selenium.webdriver.common.by import By # By.NAME
from selenium.webdriver.common.keys import Keys # 엔터키 사용(Keys.ENTER) 
from bs4 import BeautifulSoup # html 파싱(find, select)
from urllib.request import urlretrieve # server image 
import os # dir 경로/생성/이동
import time

def celeb_crawler(name) :    
    # 1. dirver 객체 생성  
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    
    # 1. 이미지 검색 url 
    driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
    
    # 2. 검색 입력상자 tag -> 검색어 입력   
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(name) # 검색어 입력     
    search_box.send_keys(Keys.ENTER)
    time.sleep(3) # 3초 대기(자원 loading)
    
    
    # ------------ 스크롤바 내림 ------------------------------------------------------ 
    last_height = driver.execute_script("return document.body.scrollHeight") #현재 스크롤 높이 계산
    
    while True: # 무한반복
        # 브라우저 끝까지 스크롤바 내리기
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 
        
        time.sleep(2) # 2초 대기 - 화면 스크롤 확인
    
        # 화면 갱신된 화면의 스크롤 높이 계산
        new_height = driver.execute_script("return document.body.scrollHeight")

        # 새로 계산한 스크롤 높이와 같으면 stop
        if new_height == last_height: 
            break
        last_height = new_height # 새로 계산한 스크롤 높이로 대체 
    #-------------------------------------------------------------------------
    
    
    # 3. 이미지 div 태그 수집  
    image_url = []
    for i in range(50) : # image 개수 지정                 
        src = driver.page_source # 현재페이지 source 수집 
        html = BeautifulSoup(src, "html.parser")
               
        # 상위태그 : <div class="wIjY0d jFk0f"> 하위태그 : div[n]
        div_img = html.select_one(f'div[class="wIjY0d jFk0f"] > div:nth-of-type({i+1})') # div 1개 수집
    
         
        # 4. img 태그 수집 & image url 추출
        img_tag = div_img.select_one('img[class="YQ4gaf"]') 
        
        try :
            image_url.append(img_tag.attrs['src']) # url 추출 
            print(str(i+1) + '번째 image url 추출')
        except :
            print(str(i+1) + '번째 image url 없음')
      
    print(image_url)        
    
    # 5. 중복 image url 삭제      
    print('중복 삭제 전 :',len(image_url)) # 44      
    image_url = list(set(image_url)) # 중복 url  삭제 
    print('중복 삭제 후 :', len(image_url)) # 22
    
       
    # 6. image 저장 폴더 생성과 이동 
    pwd = r'C:\ITWILL\7_Tensorflow\workspace' # base 저장 경로 
    os.mkdir(pwd + '/' + name) # 폴더 만들기(셀럽 이) 
    os.chdir(pwd + '/' + name) # 폴더 이동
        
    # 7. image url -> image save
    for i in range(len(image_url)) :
        try : # 예외처리 : server file 없음 예외처리 
            file_name = "test"+str(i+1)+".jpg" # test1.jsp
            # server image -> file save
            urlretrieve(image_url[i], filename=file_name)#(url, filepath)
            print(str(i+1) + '번째 image 저장')
        except :
            print('해당 url에 image 없음 : ', image_url[i])        
            
    driver.close() # driver 닫기 
    
   
# 1차 테스트 함수 호출 
'''
celeb_crawler("하정우")   

'''
# 2차 테스트 : 여러명 셀럽 이미지 수집  
namelist = ["조인성", "송강호", "전지현"] # 32 29, 30

for name in namelist :
    celeb_crawler(name) # image crawling

###########################################################################################
    
"""
image 얼굴인식과 68 point landmark 인식

- 필요한 패키지 설치와 68 point landmark data 준비(ppt 참고)
 1. cmake 설치 : dlib 의존성 패키지 설치 
 (tensorflow) >pip install cmake

 2. dlib 설치 : 68 랜드마크 얼굴인식
 (tensorflow) >conda install 파일경로/dlib
    
 3. scikit-image 설치 : image read/save
 (tensorflow) >pip install scikit-image

 4. 68 point landmark data 다운로드    
"""

import dlib # face detection
from skimage import io # image read/save
from glob import glob # dir 패턴검색(*jpg)

# 이미지 파일 경로 지정  
path = r'C:\ITWILL\7_Tensorflow\workspace\chap06_Face_detection'
image_path = path + "/images" # 작업 대상 image 위치  

# 1. 얼굴 인식기
face_detector = dlib.get_frontal_face_detector()

# 2. 얼굴 68 point landmark 객체 생성 
path = r'C:\ITWILL\7_Tensorflow\tools'
face_68_landmark = dlib.shape_predictor(path+'/shape_predictor_68_face_landmarks.dat')


# 3. 이미지 폴더에서 한 장씩 이미지 인식 
for file in glob(image_path+"/*.jpg") : # 폴더에서 순서대로 jpg 파일 읽기 
    image = io.imread(file) # image file 읽기 
    print(image.shape) # image 모양 
    
    # 1) 윈도에 image 표시 
    win = dlib.image_window() # 이미지 윈도 
    win.set_image(image) # 윈도에 원본 이미지 표시 
    
    # 2) image에서 얼굴인식     
    faces = face_detector(image, 1) # 두번째 인수=1 : 업샘플링 횟수 
    print('인식한 face size =', len(faces)) # 인식된 얼굴 개수 
    
    # 3) 이미지 위에 얼굴 사각점 표시  
    for face in faces : # n명 -> n번 반복
        
        print(face) # 얼굴 사각점 : [(141, 171) (409, 439)]-      
        print(f'왼쪽 : {face.left()}, 위 : {face.top()}, 오른쪽 : {face.right()}, 아래 : {face.bottom()}')
                
        # 단계1 : 윈도에 인식된 얼굴 표시 : face 사각점 좌표 겹치기 
        win.add_overlay(face) # 2차 : 이미지 위에 얼굴 겹치기 
        
        # 단계2 : face 사각점에서 68 point 겹치기
        face_landmark = face_68_landmark(image, face)
        win.add_overlay(face_landmark) # 3차 : 68 포인트 겹치기 
        
        rect = face_landmark.rect # 사각좌표[(좌,상),(우,하)] 
        
        # 단계3 : 사각좌표 기준 자르기(크롭 :crop)  : 얼굴 부분만 자르기 : image[상:하, 좌:우]
        crop = image[rect.top():rect.bottom(), rect.left():rect.right()]
        
        # 단계4 : 크롭 이미지 저장 
        io.imsave(image_path + "/croped"+str(1)+".jpg", crop) 
       
###########################################################################################

"""
croped image resize(100x100)
"""

from glob import glob # (*.jpg)
from PIL import Image # image file read
import numpy as np

# 폴더 경로 
path = r'C:\ITWILL\7_Tensorflow\workspace\chap06_Face_detection'
image_path = path + "/croped_images" # image path 


# 이미지 크기 규격화 함수 
def imgReshape() :     
    img_reshape = [] # image save 
    
    for file in glob(image_path + "/*.jpg") :  
        img = Image.open(file) # image read 
        
        # image 규격화 
        img = img.resize( (100, 100) )
        
        # PIL -> numpy
        img_data = np.array(img)    
        img_reshape.append(img_data)
    
    return np.array(img_reshape) # list -> numpy

# 함수 호출         
img_reshape = imgReshape()    
    
###########################################################################################

"""
 1. celeb5 이미지 분류기 : CNN 
 2. Image Generator : Model 공급할 이미지 생성
"""

from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution layer
from tensorflow.keras.layers import Dense, Flatten # DNN layer
import os

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)

# image resize
img_h = 150 # height
img_w = 150 # width
input_shape = (img_h, img_w, 3) # input image 

# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 : [5,5,3,32]
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 : [3,3,32,64]
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


# Flatten layer : 3d -> 1d
model.add(Flatten()) # 전결합층 

# DNN hidden layer(Fully connected layer)
model.add(Dense(64, activation = 'relu'))

# DNN Output layer
model.add(Dense(5, activation = 'softmax')) # 5 classes 

# model training set : Adam or RMSprop 
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',  
              metrics = ['accuracy'])


# 2. image file preprocessing : ImageDataGenerator 이용  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 경로 
base_dir = r"C:\ITWILL\7_Tensorflow\workspace\chap06_Face_detection"

train_dir = os.path.join(base_dir, 'images_celeb5/train_celeb5') 
validation_dir = os.path.join(base_dir, 'images_celeb5/val_celeb5')


# 특정 폴더의 이미지 분류를 위한 학습 데이터셋 생성기
train_data = ImageDataGenerator(rescale=1./255) # 정규화 

# 특정 폴더의 이미지 분류를 위한 검증 데이터셋 생성기
validation_data = ImageDataGenerator(rescale=1./255) # 정규화 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150), # image reshape
        batch_size=20, # batch size
        class_mode='categorical') # categorical label

validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='categorical')# categorical label


# 3. model training : ImageDataGenerator 이용 모델 훈련 
model_fit = model.fit_generator(
          generator = train_generator, 
          steps_per_epoch=50, 
          epochs=10, 
          validation_data=validation_generator,
          validation_steps=13) 

# model evaluation
print('model evaluation')
model.evaluate(validation_generator)


# 4. model history graph
import matplotlib.pyplot as plt
import os # os 환경설정 : 시각화 오류 해결 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
print(model_fit.history.keys())

loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']


## 과적합 시작점 확인  
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

###########################################################################################

"""
과적합 해결 방안 적용 
 - Dropout 적용 
 - EarlyStopping
 - epoch size 증가  
"""

from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution layer
from tensorflow.keras.layers import Dense, Flatten, Dropout  # [추가] DNN layer
from tensorflow.keras.callbacks import EarlyStopping # [추가]
import os

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)

# image resize
img_h = 150 # height
img_w = 150 # width
input_shape = (img_h, img_w, 3) # input image 

# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 : [5,5,3,32]
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))


# Convolution layer2 : [3,3,32,64]
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


# Flatten layer : 3d -> 1d
model.add(Flatten()) # 전결합층 

model.add(Dropout(rate=0.5)) # [추가]

# DNN hidden layer(Fully connected layer)
model.add(Dense(64, activation = 'relu'))

# DNN Output layer
model.add(Dense(5, activation = 'softmax')) # 5명 분류 

# model training set : Adam or RMSprop 
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])


# 2. image file preprocessing : image 제너레이터 이용  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 경로 지정 
base_dir = r"C:\ITWILL\7_Tensorflow\workspace\chap06_Face_detection"

train_dir = os.path.join(base_dir, 'images_celeb5/train_celeb5') 
validation_dir = os.path.join(base_dir, 'images_celeb5/val_celeb5')


# 특정 폴더의 이미지 분류를 위한 학습 데이터셋 생성기
train_data = ImageDataGenerator(rescale=1./255) # 정규화 

# 특정 폴더의 이미지 분류를 위한 검증 데이터셋 생성기
validation_data = ImageDataGenerator(rescale=1./255) # 정규화 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150), # image reshape
        batch_size=20, # batch size
        class_mode='categorical') 
# Found 990 images belonging to 5 classes.

validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='categorical')
# Found 250 images belonging to 5 classes.


# 3. model training : image제너레이터 이용 모델 훈련 
callback = EarlyStopping(monitor='val_loss', patience=2) # [추가]
# epoch=2 연속으로 검증 손실이 개선되지 않으면 조기종료 

model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=50, 
          epochs=20, # [수정] 
          validation_data=validation_generator,
          validation_steps=13,
          callbacks = [callback]) # [추가]


# model evaluation
model.evaluate(validation_generator)


# 4. model history graph
import matplotlib.pyplot as plt
 
loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']


epochs = range(1, len(acc) + 1) # epochs size 

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

    
    






