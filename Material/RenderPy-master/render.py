from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector

# Additional libraries
import cv2
import math
import numpy as np
import pandas as pd
import time

# This code has been modified from the original code #################################################################
# In the original code, the structure was not clear and it was hard to understand the code ###########################
# Therefore, I have modified the code to make it easier to understand and follow by provide different functions ######
# I have also added some comments to original code and my current code ###############################################
# The aim of that is to firstly for me to understand the code before it started the project ##########################
# Secondly, it also helps me to debug the code, and check the code is working correctly or not #######################
# I have done lots of researches and self-studying on this project, some maybe not correct but it is my best attempt #
# By the way, the code would run about 15 hours to get the final result ##############################################
# I have done all the questions except the Q5.2, I tried lots of methods on adding new VR headset but failed #########
# Hope you enjoy the code and best wishes for you ####################################################################

# Init image
width = 500
height = 300
image = Image(width, height, Color(255, 255, 255, 255))

# Init z-buffer, used for depth testing when drawing triangles
# depth testing is done by comparing the z-coordinate of the pixel with the z-coordinate of the pixel in the z-buffer

zBuffer = [-float('inf')] * width * height

# Load the model and normalize it to fit in the screen
# normalizeGeometry() will scale the model to fit in the screen
model = Model('data/headset.obj')
model.normalizeGeometry()

# data path, if you are in windows, change the path to the data folder
# need to change / to \ in windows
IMU_data_path = '../IMUData.csv'

# Problem 1.3: Implement he basic transformation matrices
# Follow by EE 267 Virtual Reality Course Notes: A Brief Overview of the Graphics Pipeline
def getTranslationMatrix(x, y, z):
    return [[1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]]

# Follow by EE 267 Virtual Reality Course Notes: A Brief Overview of the Graphics Pipeline
def getScaleMatrix(x, y, z):
    return [[x, 0, 0, 0],
			[0, y, 0, 0],
			[0, 0, z, 0],
			[0, 0, 0, 1]]

# Follow by EE 267 Virtual Reality Course Notes: A Brief Overview of the Graphics Pipeline
def getRotationMatrixX(angle):
	return [[1, 0, 0, 0],
		[0, math.cos(angle), -math.sin(angle), 0],
		[0, math.sin(angle), math.cos(angle), 0],
		[0, 0, 0, 1]]

# Follow by EE 267 Virtual Reality Course Notes: A Brief Overview of the Graphics Pipeline
def getRotationMatrixY(angle):
    return [[math.cos(angle), 0, math.sin(angle), 0],
            [0, 1, 0, 0],
            [-math.sin(angle), 0, math.cos(angle), 0],
            [0, 0, 0, 1]]

# Follow by EE 267 Virtual Reality Course Notes: A Brief Overview of the Graphics Pipeline
def getRotationMatrixZ(angle):
	return [[math.cos(angle), -math.sin(angle), 0, 0],
		[math.sin(angle), math.cos(angle), 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]]

# Matrix multiplication bu using numpy
# https://www.educative.io/blog/numpy-matrix-multiplication
def matrixMultiplication(matrix, vector):
    matrix_np = np.array(matrix)
    vector_np = np.array([vector.x, vector.y, vector.z, 1])
    result = matrix_np.dot(vector_np)
    return Vector(result[0], result[1], result[2])

# Problem 2.1: Read and import provided dataset
def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

# Problem 2.2: Convert rotational rate to radians
def convert_rate_to_radians(gyroscope):
    return gyroscope * np.pi / 180

# normalize the magnitude of the vectors, different from the one in the previous function that take data frame as input
def normalize_magnitude(vectorx, vectory, vectorz):
	magnitude = np.sqrt(vectorx**2 + vectory**2 + vectorz**2)
	# take care of the case where the magnitude is 0 (handle NAN case)
	if magnitude == 0:
		return vectorx, vectory, vectorz
	return vectorx/magnitude, vectory/magnitude, vectorz/magnitude

# normalize the magnitude of the vectors, take data frame as input
def df_normalize_magnitude(data):
    vectorx, vectory, vectorz = data
    magnitude = np.sqrt(vectorx**2 + vectory**2 + vectorz**2)
    # take care of the case where the magnitude is 0
    if magnitude == 0:
        return data
    return data/magnitude

# Problem 2.2: Prepare data
def prepare_data(data_path):
	data = read_data(data_path)
	data[' gyroscope.X'] = convert_rate_to_radians(data[' gyroscope.X'])
	data[' gyroscope.Y'] = convert_rate_to_radians(data[' gyroscope.Y'])
	data[' gyroscope.Z'] = convert_rate_to_radians(data[' gyroscope.Z'])
 
	data[[' accelerometer.X', ' accelerometer.Y', ' accelerometer.Z']] = data[[' accelerometer.X', ' accelerometer.Y', ' accelerometer.Z']].apply(df_normalize_magnitude, axis=1, result_type='broadcast')
	data[[' magnetometer.X', ' magnetometer.Y', ' magnetometer.Z ']] = data[[' magnetometer.X', ' magnetometer.Y', ' magnetometer.Z ']].apply(df_normalize_magnitude, axis=1, result_type='broadcast')
	return data

# Problem 2.3a: Convert Euler angle readings (radians) to quaternions
# roll angle: rotation about the x-axis
# pitch angle: rotation about the y-axis
# yaw angle: rotation about the z-axis
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def euler_to_quaternion(roll_angle, pitch_angle, yaw_angle):
    w = np.cos(yaw_angle * 0.5) * np.cos(pitch_angle * 0.5) * np.cos(roll_angle * 0.5) + np.sin(yaw_angle * 0.5) * np.sin(pitch_angle * 0.5) * np.cos(roll_angle * 0.5)
    x = np.cos(yaw_angle * 0.5) * np.cos(pitch_angle * 0.5) * np.sin(roll_angle * 0.5) - np.sin(yaw_angle * 0.5) * np.sin(pitch_angle * 0.5) * np.cos(roll_angle * 0.5)
    y = np.sin(yaw_angle * 0.5) * np.cos(pitch_angle * 0.5) * np.sin(roll_angle * 0.5) + np.cos(yaw_angle * 0.5) * np.sin(pitch_angle * 0.5) * np.cos(roll_angle * 0.5)
    z = np.sin(yaw_angle * 0.5) * np.cos(pitch_angle * 0.5) * np.cos(roll_angle * 0.5) - np.cos(yaw_angle * 0.5) * np.sin(pitch_angle * 0.5) * np.sin(roll_angle * 0.5)
    result = np.array([w, x, y, z])
    return result

# Problem 2.3b: Convert quaternions to Euler angles (radians)
# https://math.stackexchange.com/questions/2975109/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
def quaternion_to_euler(w,x,y,z):
	roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
	pitch = np.arcsin(2 * (w * y - z * x))
	yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
	result = np.array([roll, pitch, yaw]) 
	return result

# Problem 2.3c: Convert a quaternion to its conjugate (inverse rotation)
# https://math.stackexchange.com/questions/4281031/conjugate-of-unit-quaternion
def quaternion_to_conjugate(w,x,y,z):
	result = np.array([w, -x, -y, -z])
	return result

# Problem 2.3d: Calculate the quaternion product of quaternion a and b
# https://automaticaddison.com/how-to-multiply-two-quaternions-together-using-python/
def calculate_quaternion_product(a,b):
    a1, a2, a3, a4 = a
    b1, b2, b3, b4 = b
    w = a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4
    x = a1 * b2 + a2 * b1 + a3 * b4 - a4 * b3
    y = a1 * b3 - a2 * b4 + a3 * b1 + a4 * b2
    z = a1 * b4 + a2 * b3 - a3 * b2 + a4 * b1
    result = np.array([w, x, y, z])
    return result

# getOrthographicProjection() will convert the vertex from world space to screen space
def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)

	return screenX, screenY

# Problem 4: DISTORTION PRE-CORRECTION
def distortion_pre_correction(x,y,c1,c2):
    
    r_u = math.sqrt(x**2 + y**2)
    r_d = r_u + c1 * r_u**3 + c2 * r_u**5
    
    theta = math.atan2(y,x)
    x_d = r_d * math.cos(theta)
    y_d = r_d * math.sin(theta)
    return x_d, y_d

# Problem 1.2: Implement getPerspectiveProjection()
# https://gamedev.stackexchange.com/questions/154273/how-do-i-get-from-orthographic-to-perspective
# the focal length is to control the field of view, we have tested and found that 2.5 is a good value
def getPerspectiveProjection(x, y, z, focalLength=2.5, c1=0.4, c2=0.4):
    
    # Convert vertex from world space to screen space
    # by dividing the x and y coordinates by the z-coordinate (Perspective projection)
    normX = x / (z + focalLength)
    normY = y / (z + focalLength)
    
    # Problem 4: DISTORTION PRE-CORRECTION
    correct_x, correct_y = distortion_pre_correction(normX,normY,c1,c2)
    
    # Convert normalized coordinates to screen coordinates
    screenX = int((correct_x + 1.0) * width / 2.0)
    screenY = int((correct_y + 1.0) * height / 2.0)

    return screenX, screenY

# getVertexNormal() will calculate the normal of a vertex
# the normal of a vertex is the average of the normals of the adjacent faces
def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])

# Calculate face normals
# face normals are used to calculate the intensity of the light on each vertex
def render_face(vertices):
	faceNormals = {}
	for face in model.faces:
		p0, p1, p2 = [vertices[i] for i in face]
		faceNormal = (p2-p0).cross(p1-p0).normalize()

		for i in face:
			if not i in faceNormals:
				faceNormals[i] = []

			faceNormals[i].append(faceNormal)
	return faceNormals

def render_vertex(vertices):
    # Calculate vertex normals
	# vertex normals are used to calculate the intensity of the light on each vertex
	vertexNormals = []
	faceNormals = render_face(vertices)
	for vertIndex in range(len(vertices)):
		vertNorm = getVertexNormal(vertIndex, faceNormals)
		vertexNormals.append(vertNorm)
	return vertexNormals

# Main rendering function
def render_frame(roll_angle=0.0, pitch_angle=0.0 ,yaw_angle=0.0 , translation_x=0.0, translation_y=0.0, translation_z=0.0, scale_x=1.0, scale_y=1.0, scale_z=1.0, focal_length=1.0, c1=0.1, c2=0.1):
    # Clear the image
	image = Image(width, height, Color(255, 255, 255, 255))

	# Clear the z-buffer
	zBuffer = [-float('inf')] * width * height
	
	# Problem 1.3: Implement the basic transformation matrices
	translation = getTranslationMatrix(translation_x, translation_y, translation_z)
	
	# Rotation matrices for Question 3: Euler Angles
	rotationX = getRotationMatrixX(roll_angle)
	rotationY = getRotationMatrixY(pitch_angle)
	rotationZ = getRotationMatrixZ(yaw_angle)
	scaling = getScaleMatrix(scale_x, scale_y, scale_z)
	
	# Modify base on the original code
	transformed_vertices = []
	for vertex in model.vertices:
		transformed_vertex = matrixMultiplication(translation, vertex)
		transformed_vertex = matrixMultiplication(rotationX, transformed_vertex)
		transformed_vertex = matrixMultiplication(rotationY, transformed_vertex)
		transformed_vertex = matrixMultiplication(rotationZ, transformed_vertex)
		transformed_vertex = matrixMultiplication(scaling, transformed_vertex)
		transformed_vertices.append(transformed_vertex)
	
	# Render the image iteratively
	for face in model.faces:
		vertexNormals = render_vertex(transformed_vertices)
  
		p0, p1, p2 = [transformed_vertices[i] for i in face]
		n0, n1, n2 = [vertexNormals[i] for i in face]

		# Define the light direction
		# The light is coming from the top left corner of the screen -> (0, 0, -1) -> (x, y, z)
		lightDir = Vector(0, 0, -1)

		# Set to true if face should be culled
		# when cull = True, the face will not be drawn
		# when cull = False, the face will be drawn
		cull = False

		# Transform vertices and calculate lighting intensity per vertex
		# The intensity of the light is calculated by taking the dot product of the normal of the vertex and the light direction
		transformedPoints = []
		for p, n in zip([p0, p1, p2], [n0, n1, n2]):
			intensity = n * lightDir

			# Intensity < 0 means light is shining through the back of the face
			# In this case, don't draw the face at all ("back-face culling")
			if intensity < 0:
				cull = True
				break

			# Problem 1.1: Use getPerspectiveProjection() instead of getOrthographicProjection()
			# screenX, screenY = getOrthographicProjection(p.x, p.y, p.z)
   			# Problem 4: DISTORTION PRE-CORRECTION, we put this in the getPerspectiveProjection function
			screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, focal_length, c1, c2)
			transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))

		# if the face is not culled, draw the face
		if not cull:
			Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw(image, zBuffer)
	return image

# Problem 5.1: Implement the simple physic: let the object fall down with gravity acceleration and air resistance
def get_weight(mass, gravity):
	return mass * gravity

# air resistance formula is from NASA
def get_drag_force(drag_coefficient, air_density, area, velocity):
    return 0.5 * drag_coefficient * air_density * area * velocity**2

# We assume that the object is falling down with a constant acceleration
# Weight is negative, drag force is positive
# Negative as the object is falling down
def get_total_force(weight, drag_force):
    return weight + drag_force

# acceleratin is going to be negative
def get_acceleration(total_force, mass):
    return total_force / mass

# We assume that the object is falling down with a 0 initial velocity
def update_velocity(velocity, acceleration, dt):
    return velocity + acceleration * dt

# We assume the object is falling down from the origin (0.0.0)
def update_position(new_position, velocity, dt):
    return new_position + velocity * dt

# Problem 3.1: Implement the simple dead reckoning without the accelerometer (gravity-based tile correction)
# as mentioned in the Q&A: first part is without the accelerometer, second part is with the accelerometer
# Follow by "Head Tracking for the Oculus Rift"

def simple_dead_reckoning_without_accelerometer(df_data, inital_position, time_step, inital_velocity, gravity):
	new_position = inital_position
	new_velocity = inital_velocity
	orientation_quaternion = np.array([1, 0, 0, 0])
	position = []
	for index, row in df_data.iterrows():

		# gyroscope readings -> radians/sec
		gyroscope_x = row[' gyroscope.X']
		gyroscope_y = row[' gyroscope.Y']
		gyroscope_z = row[' gyroscope.Z']
  
		# convert the gyroscope readings to radians -> radians/sec * sec = radians
		angle = np.array([gyroscope_x, gyroscope_y, gyroscope_z]) * time_step

        # convert the angle to a quaternion
		rotation_quaternion = euler_to_quaternion(angle[0], angle[1], angle[2])

		# update the orientation quaternion by multiplying the current orientation quaternion by the rotation quaternion
		orientation_quaternion = calculate_quaternion_product(orientation_quaternion, rotation_quaternion)

		# convert the orientation quaternion to euler angles
		new_euler_angles = quaternion_to_euler(orientation_quaternion[0], orientation_quaternion[1], orientation_quaternion[2], orientation_quaternion[3])

		# gravity-based tile correction (without accelerometer) -> v = v + g * dt
		new_velocity += gravity * time_step
  
		# re-evaluate that position based on an estimated speed over the elapsed time -> x = x + v * dt
		new_position += new_velocity * time_step

		position.append(new_position)
  
	return position

# Problem 3.2: Implement the simple dead reckoning with the accelerometer (gravity-based tile correction)

# calculate the global frame acceleration
# Follow by "Head Tracking for the Oculus Rift" a' = q^-1 * a * q
def accelaration_to_global(accelaration_x, accelaration_y, accelaration_z,orientation_quaternion):
    
    # get the acceleration quaternion
	accelaration_quaternion = np.array([0, accelaration_x, accelaration_y, accelaration_z])
 
    # first calculate: a * q
	accelaration_quaternion = calculate_quaternion_product(accelaration_quaternion, orientation_quaternion)
	# then calculate q^-1
	conjugate_orientation_quaternion = quaternion_to_conjugate(orientation_quaternion[0], orientation_quaternion[1], orientation_quaternion[2], orientation_quaternion[3])
	# finally calculate: q^-1 * (a * q)
	acceleration_global_frame = calculate_quaternion_product(accelaration_quaternion, conjugate_orientation_quaternion)
	return acceleration_global_frame

# calculate the tilt axis
# follow by "Head Tracking for the Oculus Rift" 
def calcualte_tilt_axis(acceleration_global_frame):
    # we first project the acceleration to the x-z plane: (a_x, 0, a_z), remember the acceleration_global_frame is in quaternion form (w,x,y,z)
	projected_acceleration = np.array([acceleration_global_frame[1],0, acceleration_global_frame[3]])
	# then we calculate the tilt axis: (a_z, 0, -a_x)
	tilt_axis = np.array([projected_acceleration[2], 0, -projected_acceleration[0]])
	# normalize the magnitude of the tilt axis
	tilt_axis = normalize_magnitude(tilt_axis[0], tilt_axis[1], tilt_axis[2])
	return tilt_axis

# helper function: calculate the tilt angle between two vectors
# https://www.cuemath.com/geometry/angle-between-vectors/
def angle_between_vectors(vector1, vector2):
    cosin_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosin_angle)
    return angle

# complementary_filter -> follow by this guideline: 
# https://drive.google.com/file/d/0B9rLLz1XQKmaLVJLSkRwMTU0b0E/view?resourcekey=0-oUq7ThstZRP9gGOzXQz9ZA
def complementary_filter(gyro_orientation, acceleration_orientation, alpha):
    return alpha * gyro_orientation + (1 - alpha) * acceleration_orientation

# Follow by "Head Tracking for the Oculus Rift"
def complete_dead_reckoning_with_accelerometer(gyroscope_x, gyroscope_y,gyroscope_z, accelerometer_X, accelerometer_Y, accelerometer_Z, time_step, alpha, orientation_quaternion):
    
	# Transform the gyroscope data to angular velocity - rad/s
	gyroscope_x = gyroscope_x
	gyroscope_y = gyroscope_y
	gyroscope_z = gyroscope_z

	# red/sec * sec = rad
	angle = np.array([gyroscope_x, gyroscope_y, gyroscope_z]) * time_step

	rotation_quaternion = euler_to_quaternion(angle[0], angle[1], angle[2])

	orientation_quaternion = calculate_quaternion_product(orientation_quaternion, rotation_quaternion)

	accelaration_x, accelaration_y, accelaration_z = accelerometer_X, accelerometer_Y, accelerometer_Z

	accelaration_global_frame = accelaration_to_global(accelaration_x, accelaration_y, accelaration_z, orientation_quaternion)

	acceleration_vector = np.array([accelaration_global_frame[1], accelaration_global_frame[2], accelaration_global_frame[3]])
	# calculate the tilt axis
	tilt_axis = calcualte_tilt_axis(accelaration_global_frame)
	# calculate the tilt angle
	tilt_angle = angle_between_vectors(acceleration_vector, np.array([0, 0, 1]))

	acceleration_orientation = np.array([np.cos(tilt_angle/2), tilt_axis[0] * np.sin(tilt_angle/2), tilt_axis[1] * np.sin(tilt_angle/2), tilt_axis[2] * np.sin(tilt_angle/2)])
	
	# calcualte the fused orientation by using complementary filter
	fused_orientation = complementary_filter(rotation_quaternion, acceleration_orientation, alpha)
 
	# convert the fused orientation to euler angles
	fused_orientation_angle = quaternion_to_euler(fused_orientation[0], fused_orientation[1], fused_orientation[2], fused_orientation[3])
 
	pitch = fused_orientation_angle[1]
	roll = fused_orientation_angle[0]
	yaw = fused_orientation_angle[2]
 
	return pitch, roll, yaw, fused_orientation

def render(new_position, new_velocity, time_stap, weight, drag_coefficient, air_density, area, c1, c2, focal_length,IMU_data_path,mass):
	data = prepare_data(IMU_data_path)
	# total count for the number of frames
	total_count = len(data)
	# count for the number of frames
	count = 0
	# initial position as Q3.1 suggests
	orientation_quaternion = np.array([1, 0, 0, 0])
 
	for index, each in data.iterrows():
		# process time
		start_time = time.time()
		# if physics not showing good, please comment line 461, 462, 464, 465
		pitch, roll, yaw, orientation_quaternion = complete_dead_reckoning_with_accelerometer(each[' gyroscope.X'], each[' gyroscope.Y'], each[' gyroscope.Z'], each[' accelerometer.X'], each[' accelerometer.Y'], each[' accelerometer.Z'], time_stap, 0.03, orientation_quaternion)
		new_velocity = update_velocity(new_velocity, get_acceleration(get_total_force(weight, get_drag_force(drag_coefficient, air_density, area, new_velocity)), mass), time_stap)
		new_position = update_position(new_position, new_velocity, time_stap)
  
		print("new_position: ", new_position)
		print("new_velocity: ", new_velocity)
		roll_angle = roll
		pitch_angle = pitch
		yaw_angle = yaw
  
		# We have tested: translation_y = new_position, translation_z = -new_position is the best performance
		frame_image = render_frame(roll_angle=roll_angle, pitch_angle=pitch_angle, yaw_angle=yaw_angle, translation_x=0.0, translation_y=new_position, translation_z=-new_position, scale_x=1.0, scale_y=1.0, scale_z=1.0, focal_length=focal_length, c1=c1, c2=c2)
		# in order to show the frame, we need to convert the frame to numpy array, get_np_array() is the new function I added into the image.py
		frame = frame_image.get_np_array()
		cv2.imshow('frame', frame)
		# Save each frame to a separate file
		frame_filename = './frames/frame_{:04d}.png'.format(count)
		cv2.imwrite(frame_filename, frame)
		end_time = time.time()
		# print the information
		print("process time: ", end_time - start_time)
		print("estimated time: ", (end_time - start_time) * (total_count - count), "s")
		print("processing frame: ", count, "/", total_count, "")
		print("roll_angle: ", roll_angle, "pitch_angle: ", pitch_angle, "yaw_angle: ", yaw_angle)
		print("--------------------------------------------------")
		print("Process: ", count/total_count*100, "%")
		count = count + 1
		if cv2.waitKey(256) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()

def main():
    
	############      Environment Setting      ############

	# We assume that the object is falling down with a 0 initial velocity
	inital_velocity = 0.0

	# We assume the gravity is consent, and the direction is down:
	inital_gravity = -9.81 # m/s^2
 
	# 256 HZ from question 2:
	frame_rate = 256
	time_stap = 1/frame_rate

	# We assume the object is Quest2, data from: https://en.wikipedia.org/wiki/Quest_2
	mass = 0.503 # kg
	weight = get_weight(mass, inital_gravity)

	# We assume the Quest2 is a half sphere, drag_coefficient data from: https://en.wikipedia.org/wiki/Drag_coefficient
	drag_coefficient = 0.42 # sphere

	# We assume the air density is consent:
	# And it will not change with the height:
	# We assume the height of the inital headset is 5km:
	# We use the air density of 5km, and data from: https://steamcommunity.com/sharedfiles/filedetails/?id=2942123847
	air_density = 0.683 # kg/m^3 

	# length and width of the headset, data from: https://www.gsmarena.com/oculus_quest_2_review-news-46255.php
	length_object = 0.1915 # m
	width_object = 0.102 # m
	# We assume the area of the headset is the area of a rectangle:
	area = width_object * length_object
 
	# inital position is 0
	new_position = 0.0
	# c1 and c2 for distortion, chose randomly
	c1 = 0.4
	c2 = 0.4
	# focal length, we test the focal length from 1,1.5,2,2.5, and chose the best one
	focal_length = 2.5

	render(new_position, inital_velocity, time_stap, weight, drag_coefficient, air_density, area, c1, c2, focal_length,IMU_data_path,mass)

main()