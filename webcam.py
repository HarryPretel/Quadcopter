import cv2

cap = cv2.VideoCapture(0);

while(True):
	ret, frame = cap.read()
	frame = cv2.rectangle(frame,(400,0),(510,128),(0,255,0),3)
	cv2.imshow('frame', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()