import cv2
import numpy as np
import matplotlib.pyplot as plt


def edge_detection(image):
    if image is None:
        print("Error: The image could not be loaded.")
        return None, None, None
    
    
    edges_roberts = cv2.Canny(image, 50, 150)
    
    
    edges_canny = cv2.Canny(image, 100, 200)
    
    
    edges_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    edges_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)

    return edges_roberts, edges_canny, edges_sobel


def image_segmentation(image):
    if image is None:
        print("Error: The image could not be loaded.")
        return None
    
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    return binary_image


image1 = cv2.imread('location/1.jpg')
image2 = cv2.imread('location/2.jpg')
image3 = cv2.imread('location/3.jpg')


if image1 is None:
    print("Error: Unable to load image1")
if image2 is None:
    print("Error: Unable to load image2")
if image3 is None:
    print("Error: Unable to load image3")


if image1 is not None:
    edges1_roberts, edges1_canny, edges1_sobel = edge_detection(image1)
    segmented_image1 = image_segmentation(image1)

if image2 is not None:
    edges2_roberts, edges2_canny, edges2_sobel = edge_detection(image2)
    segmented_image2 = image_segmentation(image2)

if image3 is not None:
    edges3_roberts, edges3_canny, edges3_sobel = edge_detection(image3)
    segmented_image3 = image_segmentation(image3)


plt.figure(figsize=(12, 12))

if image1 is not None:
    plt.subplot(3, 4, 1), plt.imshow(edges1_roberts, cmap='gray'), plt.title('Edges (Roberts) 1')
    plt.subplot(3, 4, 2), plt.imshow(edges1_canny, cmap='gray'), plt.title('Edges (Canny) 1')
    plt.subplot(3, 4, 3), plt.imshow(edges1_sobel, cmap='gray'), plt.title('Edges (Sobel) 1')
    plt.subplot(3, 4, 4), plt.imshow(segmented_image1, cmap='gray'), plt.title('Segmentation 1')

if image2 is not None:
    plt.subplot(3, 4, 5), plt.imshow(edges2_roberts, cmap='gray'), plt.title('Edges (Roberts) 2')
    plt.subplot(3, 4, 6), plt.imshow(edges2_canny, cmap='gray'), plt.title('Edges (Canny) 2')
    plt.subplot(3, 4, 7), plt.imshow(edges2_sobel, cmap='gray'), plt.title('Edges (Sobel) 2')
    plt.subplot(3, 4, 8), plt.imshow(segmented_image2, cmap='gray'), plt.title('Segmentation 2')

if image3 is not None:
    plt.subplot(3, 4, 9), plt.imshow(edges3_roberts, cmap='gray'), plt.title('Edges (Roberts) 3')
    plt.subplot(3, 4, 10), plt.imshow(edges3_canny, cmap='gray'), plt.title('Edges (Canny) 3')
    plt.subplot(3, 4, 11), plt.imshow(edges3_sobel, cmap='gray'), plt.title('Edges (Sobel) 3')
    plt.subplot(3, 4, 12), plt.imshow(segmented_image3, cmap='gray'), plt.title('Segmentation 3')

plt.show()
