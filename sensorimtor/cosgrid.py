import numpy as np
import cv2 as cv

clicked = True

def mouse(event, pos_x, pos_y, flags, param):
	global x, key, prev_grid_cell_basis, clicked
	if event == cv.EVENT_LBUTTONDOWN:
		clicked = True
	elif event == cv.EVENT_LBUTTONUP:
		clicked = False
	if not clicked:
		return
	prev_x = x.copy()
	x[0], x[1] = pos_x // 10, pos_y // 10
	if key == ord('w'):
		x[1] -= 1
	elif key == ord('a'):
		x[0] -= 1
	elif key == ord('s'):
		x[1] += 1
	elif key == ord('d'):
		x[0] += 1
	
	x[0], x[1] = x[0] % 100, x[1] % 100
	if (x == prev_x).all():
		return
	cartesian = np.full((100, 100), 255, dtype=np.uint8)
	cartesian[x[1], x[0]] = 0
	cartesian= cv.resize(cartesian, (cartesian.shape[1] * 10, cartesian.shape[0] * 10), interpolation=cv.INTER_NEAREST)
	cv.imshow('cartesian', cartesian)

	grid_cell_basis = W @ x + b
	grid_cell_module = np.cos(np.array([grid_cell_basis - i / 10.0 * 2.0 * np.pi for i in range(10)])) ** 2
	grid_cell_module = cv.resize(grid_cell_module, (grid_cell_module.shape[1] * 100, grid_cell_module.shape[0] * 100), interpolation=cv.INTER_NEAREST)
	cv.imshow('grid_cell_module', grid_cell_module)

	displacement = grid_cell_basis - prev_grid_cell_basis
	disp_cell_module = np.cos(np.array([displacement - i / 10.0 * 2.0 * np.pi for i in range(10)])) ** 2
	disp_cell_module = cv.resize(disp_cell_module, (disp_cell_module.shape[1] * 100, disp_cell_module.shape[0] * 100), interpolation=cv.INTER_NEAREST)
	cv.imshow('disp_cell_module', disp_cell_module)
	prev_grid_cell_basis = grid_cell_basis

cv.namedWindow('cartesian')
cv.setMouseCallback('cartesian', mouse)

x = np.array([0, 0], dtype=np.int)
prev_grid_cell_basis = np.zeros(4)
W = np.random.randn(4, 2)
b = np.random.randn(4)

key = 0
while key != ord('q'):
	key = cv.waitKey(1) & 0xff

'''
key = 0
while key != ord('q'):
	if key == ord('w'):
		x[1] -= 1
	elif key == ord('a'):
		x[0] -= 1
	elif key == ord('s'):
		x[1] += 1
	elif key == ord('d'):
		x[0] += 1
	
	x[0], x[1] = x[0] % 100, x[1] % 100
	cartesian = np.full((100, 100), 255, dtype=np.uint8)
	cartesian[x[1], x[0]] = 0
	cartesian= cv.resize(cartesian, (cartesian.shape[1] * 10, cartesian.shape[0] * 10), interpolation=cv.INTER_NEAREST)
	cv.imshow('cartesian', cartesian)

	grid_cell_basis = W @ x + b
	grid_cell_module = np.cos(np.array([grid_cell_basis - i / 10.0 * 2.0 * np.pi for i in range(10)])) ** 2
	grid_cell_module = cv.resize(grid_cell_module, (grid_cell_module.shape[1] * 100, grid_cell_module.shape[0] * 100), interpolation=cv.INTER_NEAREST)
	cv.imshow('grid_cell_module', grid_cell_module)
	key = cv.waitKey(1) & 0xff
'''