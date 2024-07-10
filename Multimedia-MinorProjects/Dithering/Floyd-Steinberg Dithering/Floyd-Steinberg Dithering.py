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

# Floyd Steinberg Dithering Algorithm
def floydsteinbergdithering(image):
    gray = grayscale(image)
    height, width = gray.shape
    dithered_img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            old_pixel = gray[y, x]
            new_pixel = 0 if old_pixel < 128 else 255
            dithered_img[y, x] = new_pixel
            error = old_pixel - new_pixel
            if x < width - 1:
                gray[y, x + 1] += int(error * 7 / 16)
            if x > 0 and y < height - 1:
                gray[y + 1, x - 1] += int(error * 3 / 16)
            if y < height - 1:
                gray[y + 1, x] += int(error * 5 / 16)
            if x < width - 1 and y < height - 1:
                gray[y + 1, x + 1] += int(error / 16)
    cv2.imwrite("FS-dithered.png", dithered_img)
    return dithered_img

image = cv2.imread('1665_girl_with_a_pearl_earring_sm.jpg')
floydsteinbergdithering(image)
