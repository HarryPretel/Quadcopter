from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt



FPS = 100	# 1/FPS seconds = time program pauses between frames
markerLength = 2 # 2 cm = phone; 10 cm = printout


#aruco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000) #markers w/ id 1-1000 may be used
arucoParams = cv2.aruco.DetectorParameters_create() 

#calibration setup
calibrationFile = "calibrationFileName.xml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ) 
camera_matrix = calibrationParams.getNode("cameraMatrix").mat() 
dist_coeffs = calibrationParams.getNode("distCoeffs").mat() 
r = calibrationParams.getNode("R").mat() 
new_camera_matrix = calibrationParams.getNode("newCameraMatrix").mat() 

#facial detection setup
face_cascade = cv2.CascadeClassifier('../../../../../Miniconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../../../../../Miniconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml')


def cameraPoseFromHomography(H): 
	H1 = H[:, 0] 
	H2 = H[:, 1] 
	H3 = np.cross(H1, H2) 

	norm1 = np.linalg.norm(H1) 
	norm2 = np.linalg.norm(H2) 
	tnorm = (norm1 + norm2) / 2.0; 

	T = H[:, 2] / tnorm 
	return np.mat([H1, H2, H3, T]) 

def draw(img, corners, imgpts): 
	imgpts = np.int32(imgpts).reshape(-1, 2) 

	# draw ground floor in green 
	img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3) 

	# draw pillars in blue color 
	for i, j in zip(range(4), range(4, 8)): 
		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3) 

	# draw top layer in red color 
	img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3) 
	return img 


class FrontEnd(object):
	""" Maintains the Tello display and moves it through the keyboard keys.
		Press escape key to quit.
		The controls are:
			- T: Takeoff
			- L: Land
			- Arrow keys: Forward, backward, left and right.
			- A and D: Counter clockwise and clockwise rotations
			- W and S: Up and down.
	"""

	def __init__(self):
		# Init pygame
		pygame.init()

		# Creat pygame window
		pygame.display.set_caption("Tello video stream")
		self.screen = pygame.display.set_mode([960, 720]) # ((width, height) of window))

		# Init Tello object that interacts with the Tello drone
		self.tello = Tello()

		# Drone velocities between -100~100
		self.for_back_velocity = 0
		self.left_right_velocity = 0
		self.up_down_velocity = 0
		self.yaw_velocity = 0
		self.speed = 10

		self.send_rc_control = False

		# create update timer
		pygame.time.set_timer(USEREVENT + 1, 50)
		


	def run(self):

		#todo: change to arrays to reduce xyz

		#X direction
		kp_x = .15; #proportional constant
		ki_x = 0;	#integral constant
		kd_x =  .05;	#derivative constant
		
		kp_y = .5;
		ki_y = 0;
		kd_y =  .1;
		
		kp_z = .4;
		ki_z = 0;
		kd_z =  .1;

		bias_x = 0;
		bias_y = 0;
		bias_z = 0;

		prev_error_x = 0;
		prev_error_y = 0;
		prev_error_z = 0;

		integral_x = 0;
		integral_y = 0;
		integral_z = 0;
		
		t = 0;	#time (used for plotting)

		plt.plot(t, kp_x, 'r--', t, kp_y, 'bs', t, kp_z, 'g^') #todo: remove
		
		plt.ion() #interactive mode on
		plt.show()
		
		plt.pause(0.001)
		t = t + 0.001;

		#desired xyz distance from aruco
		desired_x = 0;
		desired_y = 0;
		desired_z = 25;
		
		#range needed within desired to cut motors + attempt landing
		x_land = 0;
		y_land = 0;
		z_land = 0;
		
		
		if not self.tello.connect():
			print("Tello not connected")
			return

		if not self.tello.set_speed(self.speed):
			print("Not set speed to lowest possible")
			return

		# In case streaming is on. This happens when we quit this program without the escape key.
		if not self.tello.streamoff():
			print("Could not stop video stream")
			return

		if not self.tello.streamon():
			print("Could not start video stream")
			return

		frame_read = self.tello.get_frame_read()

		should_stop = False
		while not should_stop:

			for event in pygame.event.get():
				if event.type == USEREVENT + 1:
					self.update()
				elif event.type == QUIT:
					should_stop = True
				elif event.type == KEYDOWN:
					if event.key == K_ESCAPE:
						should_stop = True
					else:
						self.keydown(event.key)
				elif event.type == KEYUP:
					self.keyup(event.key)

			if frame_read.stopped:
				frame_read.stop()
				break

			self.screen.fill([0, 0, 0])
			frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
			
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			size = img.shape
			
			
			avg1 = np.float32(gray) 
			avg2 = np.float32(gray) 
			res = cv2.aruco.detectMarkers(gray, aruco_dict, parameters = arucoParams) 
			imgWithAruco = gray # assign imRemapped_color to imgWithAruco directly 
			#if len(res[0]) > 0: 
				#print (res[0]) 
			

			focal_length = size[1] 
			center = (size[1]/2, size[0]/2) 
			camera_matrix = np.array( 
							[[focal_length, 0, center[0]], 
							[0, focal_length, center[1]], 
							[0, 0, 1]], dtype = "double"
							) 
			
			if res[1] != None: # if aruco marker detected 
				im_src = imgWithAruco 
				im_dst = imgWithAruco 
		
				pts_dst = np.array([[res[0][0][0][0][0], res[0][0][0][0][1]], [res[0][0][0][1][0], res[0][0][0][1][1]], [res[0][0][0][2][0], res[0][0][0][2][1]], [res[0][0][0][3][0], res[0][0][0][3][1]]]) 
				pts_src = pts_dst 
				h, status = cv2.findHomography(pts_src, pts_dst) 

				imgWithAruco = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0])) 

				rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(res[0], markerLength, camera_matrix, dist_coeffs) 
				img = cv2.aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec, tvec, 10) 
				cameraPose = cameraPoseFromHomography(h) 
				
				#PID controller
				x = tvec[0][0][0];
				y = tvec[0][0][1];
				z = tvec[0][0][2];
				
				plt.subplot(311)
				plt.plot(t, x, 'rs')
				plt.subplot(312)
				plt.plot(t, y, 'gs');
				plt.subplot(313)
				plt.plot(t, z, 'bs');
				
				iteration_time = 1/FPS;
				
				error_x = desired_x - x;
				integral_x = integral_x + (error_x * iteration_time );
				derivative_x = (error_x - prev_error_x) / iteration_time;
				output_x = kp_x*error_x + ki_x*integral_x + kd_x*derivative_x + bias_x;
				prev_error_x = error_x;
				
				error_y = desired_y - y;
				integral_y = integral_y + (error_y * iteration_time );
				derivative_y = (error_y - prev_error_y) / iteration_time;
				output_y = kp_y*error_y + ki_y*integral_y + kd_y*derivative_y + bias_y;
				prev_error_y = error_y;
				
				error_z = desired_z - z;
				integral_z = integral_z + (error_z * iteration_time );
				derivative_z = (error_z - prev_error_z) / iteration_time;
				output_z = kp_z*error_z + ki_z*integral_z + kd_z*derivative_z + bias_z;
				prev_error_z = error_z;
				
				#speed can only be 10-100, if outside of range, set to 10 or 100
				if output_x>100:
					output_x = 100;
				elif output_x<-100:
					output_x = -100;
				elif output_x<10 and output_x>0:
					output_x = 10;
				elif output_x>-10 and output_x<0:
					output_x = -10;
				
				if output_y>100:
					output_y = 100;
				elif output_y<-100:
					output_y = -100;
				elif output_y<10 and output_y>0:
					output_y = 10;
				elif output_y>-10 and output_y<0:
					output_y = -10;
				
				if output_z>100:
					output_z = 100;
				elif output_z<-100:
					output_z = -100;
				elif output_z<10 and output_z>0:
					output_z = 10;
				elif output_z>-10 and output_z<0:
					output_z = -10;
				
					
				self.left_right_velocity = int(-output_x);
				self.yaw_velocity = int(-output_x);
				
				self.up_down_velocity = int(output_y);
				self.for_back_velocity = int(-output_z);
				
				if x < desired_x + x_land and x > desired_x - x_land and y < desired_y + y_land and y > desired_y - y_land and z < desired_z + z_land and z > desired_z - z_land:
					self.tello.emergency()
					self.send_rc_control = False
				
				print("x: ",x,"y:",y,"z:",z,"dz",-output_z,sep="\t")
				
				#consider using gain scheduling (different sets of terms based on different speeds
				
				
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in faces:
				img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				roi_gray = gray[y:y+h, x:x+w]
				roi_color = img[y:y+h, x:x+w]
				eyes = eye_cascade.detectMultiScale(roi_gray)
				for (ex,ey,ew,eh) in eyes:
					cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			
			
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			frame = np.rot90(img)
			frame = np.flipud(frame)
			
			frame = pygame.surfarray.make_surface(frame)
			self.screen.blit(frame, (0, 0))
			pygame.display.update()
			
			#xyz graphing
			plt.show()
			plt.pause(1 / FPS)
			t = t + 1/FPS;

		#deallocate resources.
		self.tello.end()
		cv2.destroyAllWindows()

	def keydown(self, key):
		""" Update velocities based on key pressed
		Arguments:
			key: pygame key
		"""
		
			

	def keyup(self, key):
		""" Update velocities based on key released
		Arguments:
			key: pygame key
		"""
		
		if key == pygame.K_t:  # takeoff
			self.tello.takeoff()
			self.send_rc_control = True
		elif key == pygame.K_l:  # land
			self.tello.land()
			self.send_rc_control = False
		elif key == pygame.K_e:		#release E to turn off all motors (for our automated landing)
			self.tello.emergency()
			self.send_rc_control = False

	def update(self):
		""" Update routine. Send velocities to Tello."""
		if self.send_rc_control:
			self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
									   self.yaw_velocity)


def main():
	frontend = FrontEnd()

	# run frontend
	frontend.run()


if __name__ == '__main__':
	main()