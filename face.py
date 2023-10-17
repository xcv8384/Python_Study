#pip install opencv-python

import cv2
import numpy as np

# 얼굴 인식 모델 불러오기 (Haar Cascade 또는 다른 모델을 사용할 수 있음)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 예시: Haar Cascade

# 합성할 이미지 불러오기
overlay_image = cv2.imread("C:/Users/user14/Downloads/dgbrdmb_setimgmake.jpg")  # 이미지 파일 경로 지정

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)  # 카메라로부터 비디오 캡처 (0은 기본 카메라)

while True:
    ret, frame = cap.read()  # 비디오 프레임 읽기
    
    if not ret:
        break
    
    # 얼굴 인식
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_roi = frame[y:y+h, x:x+w]
        
        # 얼굴에 이미지 합성
        overlay_image = cv2.resize(overlay_image, (w, h))
        # 얼굴에 이미지 합성
        alpha = 0.01  # 조정 가능한 가중치 값 (0 ~ 1 사이)
        frame[y:y+h, x:x+w] = cv2.addWeighted(face_roi, alpha, overlay_image, 1 - alpha, 0)

        #frame[y:y+h, x:x+w] = cv2.addWeighted(face_roi, 1, overlay_image, 0.5, 0)
    
    # 화면에 출력
    cv2.imshow('Face Recognition', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()