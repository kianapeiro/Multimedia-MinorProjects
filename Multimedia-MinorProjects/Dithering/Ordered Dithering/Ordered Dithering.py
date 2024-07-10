import cv2
import numpy as np

# Grayscale conversion
def grayscale(image):
    if len(image.shape) == 3:
        height, width, channels = image.shape
        gray_image = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                blue, green, red = image[y, x]
                gray_value = int(0.299*red+0.587*green+0.114*blue)
                gray_image[y, x] = gray_value
        cv2.imwrite("grayscale.png", gray_image)  
        return gray_image
    else:
        return image 

# Ordered Dithering Algorithm
def ordereddithering(image, matrix_size):
    gray = grayscale(image)
    dither_matrix = np.array([[0, 8, 2, 10],
                              [12, 4, 14, 6],
                              [3, 11, 1, 9],
                              [15, 7, 13, 5]])
    height, width = gray.shape
    dithered_img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixel = gray[y, x]
            threshold = dither_matrix[y % matrix_size][x % matrix_size] * (256 // (matrix_size ** 2))
            # Convert pixel to black or white
            if pixel > threshold:
                dithered_img[y, x] = 255
            else:
                dithered_img[y, x] = 0
    cv2.imwrite("o_dithered.png", dithered_img)
    return dithered_img

image = cv2.imread('s.jpg')
ordereddithering(image, 4)
