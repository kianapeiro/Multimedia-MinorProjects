import cv2
import numpy as np
from heapq import heapify, heappop, heappush
from collections import Counter

# Read and convert image to ndarray
def read_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"Image at {file_path} not found")
    return image

file_path = "image.jpg"
image = read_image(file_path)
print("Image shape:", image.shape)

# Convert RGB to YCbCr
def rgb_to_ycbcr(image):
    img = image.astype(float)
    Y = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    Cb = 128 - 0.168736 * img[:,:,2] - 0.331264 * img[:,:,1] + 0.5 * img[:,:,0]
    Cr = 128 + 0.5 * img[:,:,2] - 0.418688 * img[:,:,1] - 0.081312 * img[:,:,0]
    return np.stack((Y, Cb, Cr), axis=-1)

# Convert RGB to HSV
def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

ycbcr_image = rgb_to_ycbcr(image)
hsv_image = rgb_to_hsv(image)
print("YCbCr Image shape:", ycbcr_image.shape)
print("HSV Image shape:", hsv_image.shape)

# Pad image 
def pad_image(image, block_size=8):
    h, w, c = image.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    return padded_image

padded_image = pad_image(ycbcr_image)
print("Padded Image shape:", padded_image.shape)

# Divide image into 8x8 blocks & apply DCT
def blockify(image, block_size=8):
    h, w, c = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(image[i:i+block_size, j:j+block_size])
    return np.array(blocks)

def dct_2d(block):
    return cv2.dct(block.astype(np.float32))

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.zeros_like(block)
        for i in range(block.shape[2]):  # Apply DCT channel-wise
            dct_block[:,:,i] = dct_2d(block[:,:,i])
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

blocks = blockify(padded_image)
dct_blocks = dct_transform(blocks)
print("DCT Blocks shape:", dct_blocks.shape)

# Quantize DCT coefficients
def generate_quantization_matrix(quality_factor):
    q_matrix_base = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                              [12, 12, 14, 19, 26, 58, 60, 55],
                              [14, 13, 16, 24, 40, 57, 69, 56],
                              [14, 17, 22, 29, 51, 87, 80, 62],
                              [18, 22, 37, 56, 68, 109, 103, 77],
                              [24, 35, 55, 64, 81, 104, 113, 92],
                              [49, 64, 78, 87, 103, 121, 120, 101],
                              [72, 92, 95, 98, 112, 100, 103, 99]])
    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor
    q_matrix = np.array([[int((scale * q_val + 50) / 100) for q_val in row] for row in q_matrix_base])
    return q_matrix

def quantize(block, q_matrix):
    return np.round(block / q_matrix[:, :, None])

def quantize_blocks(blocks, quality_factor):
    q_matrix = generate_quantization_matrix(quality_factor)
    return np.array([quantize(block, q_matrix) for block in blocks])

quality_factor = 50
quantized_blocks = quantize_blocks(dct_blocks, quality_factor)
print("Quantized Blocks shape:", quantized_blocks.shape)

# Huffman coding
class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(symbols):
    heap = [HuffmanNode(sym, freq) for sym, freq in symbols.items()]
    heapify(heap)
    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        merged = HuffmanNode(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heappush(heap, merged)
    return heap[0]

def generate_huffman_codes(node, code=""):
    if node.symbol is not None:
        return {node.symbol: code}
    codes = {}
    if node.left:
        codes.update(generate_huffman_codes(node.left, code + "0"))
    if node.right:
        codes.update(generate_huffman_codes(node.right, code + "1"))
    return codes

def huffman_encode(data, codes):
    return ''.join(codes[symbol] for symbol in data)

def huffman_decode(encoded_data, root):
    current = root
    decoded = []
    for bit in encoded_data:
        current = current.left if bit == '0' else current.right
        if current.symbol is not None:
            decoded.append(current.symbol)
            current = root
    return decoded

flattened_coeffs = quantized_blocks.flatten()
freq = Counter(flattened_coeffs)
huffman_tree = build_huffman_tree(freq)
huffman_codes = generate_huffman_codes(huffman_tree)
encoded_data = huffman_encode(flattened_coeffs, huffman_codes)
print("Encoded Data length:", len(encoded_data))

# Decode and dequantize
def decode_and_dequantize(encoded_data, huffman_tree, quality_factor, image_shape):
    decoded_coeffs = huffman_decode(encoded_data, huffman_tree)
    quantized_blocks = np.array(decoded_coeffs).reshape(-1, 8, 8, 3)
    q_matrix = generate_quantization_matrix(quality_factor)
    dequantized_blocks = np.array([dequantize(block, q_matrix) for block in quantized_blocks])
    return dequantized_blocks

def dequantize(block, q_matrix):
    return block * q_matrix[:, :, None]

dequantized_blocks = decode_and_dequantize(encoded_data, huffman_tree, quality_factor, padded_image.shape)
print("Dequantized Blocks shape:", dequantized_blocks.shape)

# Apply inverse DCT and inverse color transform
def idct_2d(block):
    return cv2.idct(block.astype(np.float32))

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = np.zeros_like(block)
        for i in range(block.shape[2]):
            idct_block[:,:,i] = idct_2d(block[:,:,i])
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)

def reconstruct_image_from_blocks(blocks, image_shape):
    block_size = 8
    h, w, c = image_shape
    image = np.zeros((h, w, c))
    k = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            image[i:i+block_size, j:j+block_size] = blocks[k]
            k += 1
    return image

def ycbcr_to_rgb(image):
    img = image.astype(float)
    R = img[:,:,0] + 1.402 * (img[:,:,2] - 128)
    G = img[:,:,0] - 0.344136 * (img[:,:,1] - 128) - 0.714136 * (img[:,:,2] - 128)
    B = img[:,:,0] + 1.772 * (img[:,:,1] - 128)
    return np.stack((B, G, R), axis=-1).astype(np.uint8)

idct_blocks = idct_transform(dequantized_blocks)
reconstructed_image = reconstruct_image_from_blocks(idct_blocks, padded_image.shape)
final_image = ycbcr_to_rgb(reconstructed_image)

# Apply Post-Processing Filtering
def apply_median_filter(image):
    return cv2.medianBlur(image, 3)

final_image = apply_median_filter(final_image)

print("Final Image shape:", final_image.shape)
cv2.imwrite("compressed_image.jpg", final_image)