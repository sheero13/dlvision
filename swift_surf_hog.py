# #sift
# import cv2
# import matplotlib.pyplot as plt

# # Load the image in grayscale
# image = cv2.imread("D:/OCT/OCT_ONE.JPG", cv2.IMREAD_GRAYSCALE)

# # Resize the image (optional, for better visualization)
# image = cv2.resize(image, (600, 400))

# # Initialize the SIFT detector
# sift = cv2.SIFT_create()

# # Detect key points and compute descriptors
# keypoints, descriptors = sift.detectAndCompute(image, None)

# # Draw the key points on the image
# image_with_keypoints = cv2.drawKeypoints(
#     image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0)
# )

# # Display the original image and the image with key points
# plt.figure(figsize=(12, 6))

# # Original Image
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap="gray")
# plt.axis("off")

# # Image with Key Points
# plt.subplot(1, 2, 2)
# plt.title("Image with Key Points (SIFT)")
# plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# #-------------------------------------------------------------------------------------------------------------------

# #surf

# import cv2
# import matplotlib.pyplot as plt

# # Load the image in grayscale
# image = cv2.imread("D:/OCT/OCT_ONE.JPG", cv2.IMREAD_GRAYSCALE)

# # Resize the image (optional, for better visualization)
# image = cv2.resize(image, (600, 400))

# # Initialize the SURF detector
# surf = cv2.xfeatures2d.SURF_create()

# # Detect key points and compute descriptors
# keypoints, descriptors = surf.detectAndCompute(image, None)

# # Draw the key points on the image
# image_with_keypoints = cv2.drawKeypoints(
#     image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0)
# )

# # Display the original image and the image with key points
# plt.figure(figsize=(12, 6))

# # Original Image
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap="gray")
# plt.axis("off")

# # Image with Key Points
# plt.subplot(1, 2, 2)
# plt.title("Image with Key Points (SURF)")
# plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# #---------------------------------------------------------------------------------------------------------------

# #Hog
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def compute_hog_descriptor(image, cell_size=(8, 8), block_size=(2, 2), nbins=9):
#     """
#     Computes the Histogram of Oriented Gradients (HOG) descriptor for a given image.

#     Parameters:
#         image: Input image (grayscale)
#         cell_size: Tuple (height, width) representing the size of each cell in pixels
#         block_size: Tuple (height, width) representing the size of blocks in terms of cells
#         nbins: Number of bins for the histogram of orientations

#     Returns:
#         HOG descriptor as a feature vector
#     """
#     # Calculate gradients in the x and y directions using Sobel filters
#     gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
#     gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    
#     # Calculate magnitude and orientation of the gradients
#     magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
#     # Normalize angles to [0, 180) for unsigned gradients
#     angle = angle % 180

#     # Image dimensions
#     h, w = image.shape

#     # Initialize the HOG descriptor vector
#     hog_descriptor = []

#     # Divide the image into cells
#     cell_h, cell_w = cell_size
#     n_cells_x = w // cell_w
#     n_cells_y = h // cell_h

#     # Compute histograms for each cell
#     histograms = np.zeros((n_cells_y, n_cells_x, nbins))
#     bin_width = 180 / nbins

#     for i in range(n_cells_y):
#         for j in range(n_cells_x):
#             # Extract the magnitude and angle for the current cell
#             cell_magnitude = magnitude[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
#             cell_angle = angle[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            
#             # Create a histogram for the current cell
#             hist = np.zeros(nbins)
#             for m, a in zip(cell_magnitude.ravel(), cell_angle.ravel()):
#                 bin_idx = int(a // bin_width)
#                 hist[bin_idx] += m
            
#             histograms[i, j, :] = hist

#     # Normalize histograms within blocks
#     block_h, block_w = block_size
#     for i in range(n_cells_y - block_h + 1):
#         for j in range(n_cells_x - block_w + 1):
#             block_hist = histograms[i:i + block_h, j:j + block_w, :].ravel()
#             block_hist /= np.sqrt(np.sum(block_hist ** 2) + 1e-6)  # L2 normalization
#             hog_descriptor.extend(block_hist)
    
#     return np.array(hog_descriptor)

# # Load and preprocess the image
# image = cv2.imread('D:/OCT/OCT_ONE.JPG', cv2.IMREAD_GRAYSCALE)
# if image is None:
#     raise ValueError("Image not found. Please provide a valid path.")

# # Compute the HOG descriptor
# hog_features = compute_hog_descriptor(image)

# # Display the original image and the feature vector size
# print(f"HOG Descriptor Length: {len(hog_features)}")
# plt.imshow(image, cmap='gray')
# plt.title("Original Image")
# plt.axis('off')
# plt.show()
