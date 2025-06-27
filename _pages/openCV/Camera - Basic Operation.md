---
title: "Camera - Basic Operation"
date: "2025-06-24"
thumbnail: "/assets/img/thumbnail/opencv.png"
---

# Code
---
**카메라 실행**
```
import cv2

# Read from the first camera device
cap = cv2.VideoCapture(0)

w = 640#1280#1920
h = 480#720#1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 비디오 파일을 위한 VideoWriter 객체 생성 (프레임 20)
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (w,h))

# 성공적으로 video device가 열렸으면 while 문 반복
while(cap.isOpened()):

    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 비디오 파일에 프레임 저장
    out.write(frame)

    # Display
    cv2.imshow("Camera",frame)

    # 1ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# 작업이 끝나면 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
```
**카메라 화면에 특정 도형 생성**
```
import cv2

# 클릭된 위치를 저장할 리스트
click_points = []

# 마우스 이벤트 콜백 함수 정의
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x,y))

# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50,50)
bottomRight = (300,300)

# 윈도우 생성 및 
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", draw_circle)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):

    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if not ret:
        break

    # Line
    cv2.line(frame, topLeft, bottomRight, (0,255,0), 5)

    # Rectangle
    cv2.rectangle(frame, [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0,0,255), 5)
    
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'me', [pt+80 for pt in topLeft], font, 2, (0,255,255), 10)

    # 클릭된 위치들에 원 그리기
    for point in click_points:
         cv2.circle(frame, point, 20, (0,255,255), 2)

    # Display
    cv2.imshow("Camera",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
```
**카메라 화면에 글씨 생성 및 BGR 조절**
```
import cv2

topLeft = (50, 50)

# 굵기 기본값
bold_yellow = 0
bold_blue = 0
bold_red = 0

# 색상 기본값
color_r = 255
color_g = 255
color_b = 255

# Callback function for the trackbar
def on_yellow_trackbar(value):
    global bold_yellow
    bold_yellow = value
def on_blue_trackbar(value):
    global bold_blue
    bold_blue = value
def on_red_trackbar(value):
    global bold_red
    bold_red = value
def on_r_trackbar(value):
    global color_r
    color_r = value
def on_g_trackbar(value):
    global color_g
    color_g = value
def on_b_trackbar(value):
    global color_b
    color_b = value

# 카메라 열기
cap = cv2.VideoCapture(0)

# 트랙바 생성
cv2.namedWindow("Camera")
cv2.createTrackbar("Yellow", "Camera", bold_yellow, 10, on_yellow_trackbar)
cv2.createTrackbar("Blue", "Camera", bold_blue, 10, on_blue_trackbar)
cv2.createTrackbar("Red", "Camera", bold_red, 10, on_red_trackbar)
cv2.createTrackbar("R", "Camera", color_r, 255, on_r_trackbar)
cv2.createTrackbar("G", "Camera", color_g, 255, on_g_trackbar)
cv2.createTrackbar("B", "Camera", color_b, 255, on_b_trackbar)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):

    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Cant't receive frame (stream end?). Exiting ...")
        break

    # Text
    cv2.putText(frame, "TEXT", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 1+bold_yellow)   
    cv2.putText(frame, "TEXT", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 1+bold_blue)
    cv2.putText(frame, "TEXT", (50,190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1+bold_red)

    # Color
    dynamic_color = (color_b, color_g, color_r)
    cv2.putText(frame, "TEXT", (50,260), cv2.FONT_HERSHEY_SIMPLEX, 2, dynamic_color, 2)

    # Display
    cv2.imshow("Camera",frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
```
# 분석
---
**카메라 화면에 특정 도형 생성**
>![alt text](<../../assets/img/opencv/perceptron/camera/스크린샷 2025-06-25 09-30-33.png>)

<br/>

**카메라 화면에 글씨 생성 및 BGR 조절**
>![alt text](<../../assets/img/opencv/perceptron/camera/스크린샷 2025-06-25 09-50-40.png>)글씨 생성<br/>
![alt text](<../../assets/img/opencv/perceptron/camera/스크린샷 2025-06-25 09-51-36.png>)굵기 조절<br/>
![alt text](<../../assets/img/opencv/perceptron/camera/스크린샷 2025-06-25 09-59-43.png>)BGR 조절