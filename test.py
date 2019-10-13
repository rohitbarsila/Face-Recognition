import cv2
import pandas as pd
import time
import faceRecognition as fr


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainDATA.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
df=pd.read_csv("StudentDetails\StudentDetails.csv")
x=str(input('Enter Location Of Students Photo (Without ["]): '))
cap = cv2.VideoCapture(x)
col_names =  ['Id','Name']
df=pd.read_csv("StudentDetails\StudentDetails.csv")
while True:
	ret,test_img=cap.read()
	#test_img = cv2.resize(test_img, (1080, 720))
	faces_detected,gray_img=fr.faceDetection(test_img)

	for face in faces_detected:
		(x,y,w,h)=face
		roi_gray=gray_img[y:y+h,x:x+w]
		Id,confidence=face_recognizer.predict(roi_gray)
		fr.draw_rect(test_img,face)
		if confidence>67:
			aa=df.loc[df['Id'] == Id]['Name'].values
			fr.put_text(test_img,str(aa),x,y)
	cv2.imshow('face detection Tutorial', test_img)
	cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()