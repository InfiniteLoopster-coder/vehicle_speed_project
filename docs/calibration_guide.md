# Camera Calibration Guide

**Version:** 1.0  
**Last Updated:** 2025-02-09

---

## 1. Introduction

Camera calibration is a critical step in the vehicle speed detection and tracking pipeline. It involves estimating the camera's intrinsic parameters (such as focal length and optical center) and its distortion coefficients, and computing a homography matrix to map image (pixel) coordinates to real-world (metric) coordinates. Accurate calibration ensures that speed estimations based on object displacement in the video are reliable.

---

## 2. Overview of the Calibration Process

The calibration process can be broken down into several key steps:

1. **Capture Calibration Images:**  
   Capture multiple images of a known calibration pattern (commonly a chessboard) at various orientations and positions.

2. **Corner Detection:**  
   Detect the inner corners of the chessboard pattern in each image using OpenCV functions.

3. **Compute Calibration Parameters:**  
   Use the detected points to calculate the camera matrix and distortion coefficients.

4. **Compute the Homography Matrix:**  
   Map the image coordinates to real-world coordinates using a set of known reference points.

5. **Validation and Testing:**  
   Validate the calibration by undistorting test images and verifying the accuracy of the transformation.

---

## 3. Detailed Calibration Steps

### 3.1 Capturing Calibration Images

- **Equipment:**  
  - Use a high-resolution camera.
  - Use a printed chessboard or another calibration pattern with known dimensions.

- **Procedure:**  
  - Take 20–30 images of the calibration pattern.
  - Ensure the pattern fills a significant portion of the frame.
  - Capture the pattern at different angles and positions to cover all parts of the image sensor.

### 3.2 Detecting Corners

- **Tools:**  
  Use OpenCV’s `cv2.findChessboardCorners` and `cv2.cornerSubPix`.

- **Example Code:**

  ```python
  import cv2
  import numpy as np

  # Chessboard dimensions (number of inner corners per row and column)
  chessboard_size = (9, 6)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # Prepare object points, e.g. (0,0,0), (1,0,0), …, (8,5,0)
  objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
  objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

  # Arrays to store object points and image points
  objpoints = []  # 3d points in real world space
  imgpoints = []  # 2d points in image plane

  # Read a calibration image
  img = cv2.imread('data/raw/calibration_image.jpg')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Find chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
  if ret:
      objpoints.append(objp)
      corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      imgpoints.append(corners2)
      cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
      cv2.imshow('Calibration', img)
      cv2.waitKey(500)
  cv2.destroyAllWindows()


#### 3.3 Calculating Camera Calibration Parameters

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

##### Example source points (pixels) from the image

pts_src = np.array([[100, 200], [400, 200], [400, 500], [100, 500]], dtype='float32')
# Corresponding destination points (real-world coordinates in meters)
pts_dst = np.array([[0, 0], [3, 0], [3, 4], [0, 4]], dtype='float32')

###### 3.4 Computing the Homography Matrix

homography_matrix, status = cv2.findHomography(pts_src, pts_dst)
print("Homography Matrix:\n", homography_matrix)

###### 3.5 Validating the Calibration

undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
