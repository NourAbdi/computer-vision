# Nour Abdi 206144750

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image_filenames=[]
for i in range(1,60):
    img =  f"D:/computer vision/puzzles/puzzle_homography_10/pieces/piece_{i}.jpg"
    image_filenames.append(img)
images = [cv2.imread(filename) for filename in image_filenames]

'''
Here's how Load the images works:

A list called image_filenames is created and initialized as an empty list.

A loop is run 36 times (because we have 36 images inside the pieces folder), with i taking on the values from 1 to 36.

Inside the loop, a string called img is created using an f-string. The f-string contains the path to an image file, with {i} used as a placeholder for the value of the loop variable.

The string img is appended to the list image_filenames.

After the loop completes, a list comprehension is used to load each image file in image_filenames using the cv2.imread function. The loaded images are stored in a list called images.

So, at the end of this code, images is a list of 36 images loaded from the image files specified in image_filenames.

'''


# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for each image
keypoints_list = []
descriptors_list = []
for image in images:
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Create a feature matcher
matcher = cv2.FlannBasedMatcher_create()

# Match keypoints and descriptors between adjacent images with ratio test
matches_list = []
for i in range(len(images)-1):
    matches = matcher.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.99 * n.distance:
            good_matches.append(m)
    matches_list.append(good_matches)




# Match keypoints and descriptors between adjacent images with ratio test

# Apply RANSAC to estimate the affine transformation between matched keypoints
homography_list = []
for matches, keypoints1, keypoints2 in zip(matches_list, keypoints_list[:-1], keypoints_list[1:]):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, ransacReprojThreshold=5.0)
    homography_list.append(M)

# Warp the images to align with the first image
height, width = images[0].shape[:2]
warped_images = []
for i in range(len(images)):
    if i == 0:
        warped_images.append(images[i])
    else:
        warped_img = cv2.warpAffine(images[i], homography_list[i-1], (width*(i+1), height))
        warped_images.append(warped_img)

# Combine the images into a panorama
panorama = np.zeros((height, width*(len(images)+1), 3), dtype=np.uint8)
panorama[:, :width] = warped_images[0]
for i in range(1, len(warped_images)):
    panorama[:, i*width:(i+1)*width] = warped_images[i][:, :width]
#cv2.imwrite('panor_affine.jpg', panorama)
# Display the panorama
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.show()
match_conf=500
stitcher = cv2.Stitcher_create()
#stitcher.setMatchConf(0.5)
(status, stitched) = stitcher.stitch(images)

# Display the stitched image
if status == cv2.STITCHER_OK:
    # Display the stitched image
    plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('panorama_planaasr_1.jpeg', stitched)
else:
    # Handle the case where stitching fails
    print('Error: Failed to stitch images')

# Save the stitched image


#SAVING THE IMAGE IN A SEPARATE FOLDER
for i in range(len(warped_images)):
    x_start = i * width
    x_end = (i + 1) * width
    y_start = 0
    y_end = height
    region = panorama[y_start:y_end, x_start:x_end]
    cv2.imwrite(f'D:/computer vision/sol/results/puzzle_homography_10/piece_{i+1}_relative.jpeg', region)