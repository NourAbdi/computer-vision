#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image_filenames=[]
for i in range(1,63):
    img =  f"D:/computer vision/puzzles/puzzle_affine_10/pieces/piece_{i}.jpg"
    image_filenames.append(img)

images = [cv2.imread(filename) for filename in image_filenames]

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for each image
keypoints_list = [sift.detectAndCompute(image, None) for image in images]

# Create a feature matcher
matcher = cv2.BFMatcher()

# Match keypoints and descriptors between adjacent images
matches_list = []
for i in range(len(images)-1):
    matches = matcher.match(keypoints_list[i][1], keypoints_list[i+1][1])
    matches_list.append(matches)
# matches_list = []
# for i in range(len(images)-1):
#     matches = matcher.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.9 * n.distance:
#             good_matches.append(m)
#     matches_list.append(good_matches)

# Apply RANSAC to estimate the affine transformation between matched keypoints
affine_list = []
for matches, keypoints1, keypoints2 in zip(matches_list, keypoints_list[:-1], keypoints_list[1:]):
    src_pts = np.float32([keypoints1[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts)
    affine_list.append(M)

# Stitch the images together using OpenCV's stitcher
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# Display the stitched image
plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
plt.show()

# Save the stitched image
cv2.imwrite('panorama_planar.jpeg', stitched)

