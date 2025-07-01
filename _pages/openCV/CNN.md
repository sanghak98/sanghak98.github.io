---
title: "CNN - CIFAR10"
date: "2025-07-01"
thumbnail: "/assets/img/thumbnail/opencv.png"
---

# Code
---
**데이터 정보**
>![alt text](../../assets/img/opencv/cnn/rmsprop/1.png)

<br/>

**CNN 모델 디자인**
>![alt text](../../assets/img/opencv/cnn/rmsprop/2.png)

<br/>

**모델 학습 정보 설정 및 학습**
>![alt text](../../assets/img/opencv/cnn/rmsprop/3.png)

<br/>

**그래프 설정**
>![alt text](../../assets/img/opencv/cnn/rmsprop/4.png)

<br/>

**데이터 수 확인**
>![alt text](../../assets/img/opencv/cnn/rmsprop/8.png)
>![alt text](../../assets/img/opencv/cnn/rmsprop/9.png)

<br/>

**분류 결과 확인**
>![alt text](../../assets/img/opencv/cnn/rmsprop/7.png)

# 분석
---
**RMSprop**
>Optimizer : RMSprop<br/>
Epochs : 20<br/>
Batch_size : 256

>![alt text](../../assets/img/opencv/cnn/rmsprop/5.png)그래프 결과

<br/>

**RMSprop _2**
>Optimizer : RMSprop<br/>
Epochs : 20<br/>
Batch_size : 128

>![alt text](../../assets/img/opencv/cnn/rmsprop/3_2.png)모델 학습<br/>
![alt text](../../assets/img/opencv/cnn/rmsprop/5_2.png)그래프 결과

<br/>

**SGD + Momentum**
>Optimizer : SGD + Momentum<br/>
(Learning_rate = 0.01, Momentum = 0.9)<br/>
Epochs : 20<br/>
Batch_size : 256

>![alt text](../../assets/img/opencv/cnn/sgd+momentum/3.png)모델 학습<br/>
![alt text](../../assets/img/opencv/cnn/sgd+momentum/5.png)그래프 결과

<br/>

**SGD + Momentum _2**
>Optimizer : SGD + Momentum<br/>
(Learning_rate = 0.01, Momentum = 0.9)<br/>
Epochs : 20<br/>
Batch_size : 128

>![alt text](../../assets/img/opencv/cnn/sgd+momentum/3_2.png)모델 학습<br/>
![alt text](../../assets/img/opencv/cnn/sgd+momentum/5_2.png)그래프 결과

<br/>

**Adam**
>Optimizer : Adam<br/>
(Learning_rate = 0.0001)<br/>
Epochs : 20<br/>
Batch_size : 256

>![alt text](../../assets/img/opencv/cnn/adam/3.png)모델 학습<br/>
![alt text](../../assets/img/opencv/cnn/adam/5.png)그래프 결과

<br/>

**Adam _2**
>Optimizer : Adam<br/>
(Learning_rate = 0.0001)<br/>
Epochs : 20<br/>
Batch_size : 128

>![alt text](../../assets/img/opencv/cnn/adam/3_2.png)모델 학습<br/>
![alt text](../../assets/img/opencv/cnn/adam/5_2.png)그래프 결과

<br/>

**Adam _3**
>Optimizer : Adam<br/>
(Learning_rate = 0.0005)<br/>
Epochs : 60<br/>
Batch_size : 256<br/>

>Learning_rate 조절<br/>
Dropout 추가<br/>
(0.15 / 0.15 / 0.3)<br/>
BatchNormalization 추가<br/>

>![alt text](../../assets/img/opencv/cnn/adam/3_4.png)모델 학습<br/>
![alt text](../../assets/img/opencv/cnn/adam/5_4.png)그래프 결과

<br/>

**Adam _4**
>Optimizer : Adam<br/>
(Learning_rate = 0.0005)<br/>
Epochs : 60<br/>
Batch_size : 128<br/>

>Learning_rate 조절<br/>
Dropout 추가<br/>
(0.15 / 0.15 / 0.3)<br/>
BatchNormalization 추가<br/>

>![alt text](../../assets/img/opencv/cnn/adam/3_3.png)모델 학습<br/>
![alt text](../../assets/img/opencv/cnn/adam/5_3.png)그래프 결과

# 고찰
---

