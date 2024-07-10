import cv2
import numpy as np

# Read image
def read_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{file_path}' not found.")
    return image

# Convert RGB image to grayscale
def rgb_to_grayscale(image):
    grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return grayscale.astype(np.uint8)

# Flatten the image
def flatten_image(image):
    return image.reshape(-1)

# Encode the image using RLE
def rle_encode(image):
    pixels = flatten_image(image)
    rle = []
    prev_pixel = pixels[0]
    count = 1
    for pixel in pixels[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            rle.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1
    rle.append((prev_pixel, count))
    return rle

# Save the RLE encoded data
def save_rle_encoded(file_path, rle):
    with open(file_path, 'w') as file:
        for value, count in rle:
            file.write(f"{value} {count}\n")

# Load the RLE encoded data
def load_rle_encoded(file_path):
    rle = []
    with open(file_path, 'r') as file:
        for line in file:
            value, count = line.strip().split()
            rle.append((int(value), int(count)))
    return rle

# Decode the RLE data
def rle_decode(rle, shape):
    pixels = []
    for value, count in rle:
        pixels.extend([value] * count)
    return np.array(pixels).reshape(shape).astype(np.uint8)

# Read & Convert RGB image to grayscale
file_path = "image 1.jpg"
try:
    image = read_image(file_path)
except FileNotFoundError as e:
    print(e)
    exit(1)

grayscale_image = rgb_to_grayscale(image)

# Encode using RLE
rle_encoded = rle_encode(grayscale_image)
save_rle_encoded("RLE_encoded.txt", rle_encoded)

# Decode the RLE encoded data
rle_loaded = load_rle_encoded("RLE_encoded.txt")
decoded_image = rle_decode(rle_loaded, grayscale_image.shape)
cv2.imwrite("RLE_decoded.jpg", decoded_image)

# Compare original and decoded images
original_image = grayscale_image
decoded_image = decoded_image

# Calculate the difference
difference = np.abs(original_image.astype(np.int32) - decoded_image.astype(np.int32))
print("Difference:", np.sum(difference))
