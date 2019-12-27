import numpy as np
import cv2 as cv

key = 0

W = np.random.randn(16, 2)
b = np.random.randn(16, 1)

samples = np.random.randint(0, 101, (2, 16))
samples_grid = np.cos(W @ samples + np.outer(b, np.ones((1, 16)))) ** 2
A = samples @ np.linalg.inv(samples_grid)

def mouse(event, pos_x, pos_y, flags, param):
	global W, b
	x = np.cos(W @ np.array([[pos_x, pos_y]]).T + b) ** 2
	y = A @ x
	print(np.array([pos_x, pos_y]))
	print(y.T)
	
cv.namedWindow('cartesian')
cv.setMouseCallback('cartesian', mouse)

while key != ord('q'):
	key = cv.waitKey(1) & 0xff