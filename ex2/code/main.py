import numpy as np
import cv2
import matplotlib.pyplot as plt

# File paths
image_filenames = ['assets/im_left.JPG', 'assets/im_right.JPG']
calibration_filenames = ['assets/K.txt', 'assets/max_disp.txt']

# Load the images
images = []
for filename in image_filenames:
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    images.append(image)

# Load calibration data
calibration_data = []
for filename in calibration_filenames:
    data = np.loadtxt(filename)
    calibration_data.append(data)

# Access the loaded data
im_left = images[0]
im_right = images[1]
K = calibration_data[0]
max_disp = calibration_data[1]

# Display the images
fig, axes = plt.subplots(1, 2)
axes[0].imshow(im_left, cmap='gray')
axes[0].set_title('Left Image')
axes[1].imshow(im_right, cmap='gray')
axes[1].set_title('Right Image')
plt.show()

# Access the calibration data
print('Camera Matrix (K):\n', K)
print('Maximum Disparity:', max_disp)

# Census Transform
def census_transform(image):
    transformed = np.zeros_like(image)
    h, w = image.shape
    for i in range(3, h - 2):
        for j in range(3, w - 2):
            patch = image[i-2:i+3, j-2:j+3]
            value = np.where(patch >= image[i, j], 1, 0)
            transformed[i, j] = np.packbits(value.flatten())[0]
    return transformed

# Compute census transform on both images
census_left = census_transform(im_left)
census_right = census_transform(im_right)

# Initialize cost volume
dmax = int(max_disp)
cost_volume = np.ones((im_left.shape[0], im_left.shape[1], dmax)) * float('inf')

# Compute cost volume
for d in range(dmax):
    shifted = np.roll(census_right, d, axis=1)
    difference = np.bitwise_xor(census_left, shifted)
    difference = difference.reshape(difference.shape + (1,))
    cost_volume[:, :, d] = np.sum(difference, axis=2)

# Winner-takes-all aggregation
disparity_map = np.argmin(cost_volume, axis=2)

# Consistency filtering
for i in range(im_left.shape[0]):
    for j in range(im_left.shape[1]):
        d = disparity_map[i, j]
        if j - d >= 0:
            if abs(disparity_map[i, j - d] - d) > 1:
                disparity_map[i, j] = 0

# Calculate depth maps
baseline = 1.0  # Adjust the baseline distance as needed
focal_length = K[0, 0]  # Assuming the focal length is in the top-left corner of K matrix
depth_left = baseline * focal_length / (disparity_map + 1e-7)  # To avoid division by zero
depth_right = baseline * focal_length / (disparity_map + 1e-7)  # To avoid division by zero

# Normalize depth maps
depth_left_normalized = (depth_left - np.min(depth_left)) / (np.max(depth_left) - np.min(depth_left))
depth_right_normalized = (depth_right - np.min(depth_right)) / (np.max(depth_right) - np.min(depth_right))

# Display the depth maps in grayscale
fig, axes = plt.subplots(1, 2)
depth_left_plot = axes[0].imshow(depth_left_normalized, cmap='gray')
axes[0].set_title('Left Depth Map')
plt.colorbar(depth_left_plot, ax=axes[0])
depth_right_plot = axes[1].imshow(depth_right_normalized, cmap='gray')
axes[1].set_title('Right Depth Map')
plt.colorbar(depth_right_plot, ax=axes[1])
plt.show()
