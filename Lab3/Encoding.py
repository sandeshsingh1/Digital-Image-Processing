import cv2
import os
from collections import Counter
import heapq

# -----------------------------
# STEP 1: Load Image
# -----------------------------

# Create output directory if it doesn't exist
os.makedirs("outputs/task3", exist_ok=True)

# Load grayscale image (pixel values 0–255)
img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

# Flatten image into a 1D array of pixel values
flat = img.flatten()

# Count frequency of each pixel value
freq = Counter(flat)

# -----------------------------
# STEP 2: Shannon–Fano Coding
# -----------------------------
def shannon_fano(symbols):
    """
    Recursive Shannon–Fano coding algorithm.
    Input: list of (symbol, frequency) pairs sorted by frequency.
    Output: dictionary {symbol: code}
    """
    # Base case: no symbols
    if len(symbols) <= 1:
        if len(symbols) == 1:
            return {symbols[0][0]: "0"}  # Single symbol gets code "0"
        else:
            return {}

    # Total frequency of all symbols
    total = sum(f for _, f in symbols)

    # Find split point (where accumulated frequency ~ half of total)
    acc = 0
    split = 0
    for i, (sym, f) in enumerate(symbols):
        acc += f
        if acc >= total / 2:
            split = i
            break

    # Recursively encode left and right halves
    left = shannon_fano(symbols[:split + 1])
    right = shannon_fano(symbols[split + 1:])

    # Prefix "0" to codes in left half, "1" to codes in right half
    for k in left:
        left[k] = "0" + left[k]
    for k in right:
        right[k] = "1" + right[k]

    # Merge and return dictionary
    codes = {}
    codes.update(left)
    codes.update(right)
    return codes


# Sort symbols by frequency (descending) for Shannon–Fano
symbols_sorted = sorted(freq.items(), key=lambda x: -x[1])
sf_codes = shannon_fano(symbols_sorted)

# Encode the image using Shannon–Fano codes
sf_bits_list = [sf_codes[p] for p in flat]
sf_encoded = ''.join(sf_bits_list)

# Save Shannon–Fano encoded data (⚠️ very large text file)
with open("outputs/task3/shannon_fano.txt", "w") as f:
    f.write(sf_encoded)


# -----------------------------
# STEP 3: Huffman Coding
# -----------------------------
class Node:
    """Node for Huffman tree"""
    def __init__(self, sym, freq):
        self.sym = sym      # symbol (pixel value)
        self.freq = freq    # frequency of symbol
        self.left = None    # left child
        self.right = None   # right child

    def __lt__(self, other):
        # Required for priority queue (heapq)
        return self.freq < other.freq


def huffman(freq):
    """
    Huffman coding algorithm.
    Input: dictionary {symbol: frequency}.
    Output: dictionary {symbol: code}
    """
    # Initialize priority queue with leaf nodes
    heap = [Node(s, f) for s, f in freq.items()]
    heapq.heapify(heap)

    # Build Huffman tree
    while len(heap) > 1:
        n1 = heapq.heappop(heap)  # least frequent
        n2 = heapq.heappop(heap)  # second least frequent

        # Create parent node
        parent = Node(None, n1.freq + n2.freq)
        parent.left = n1
        parent.right = n2

        heapq.heappush(heap, parent)

    # Root of Huffman tree
    root = heap[0]
    codes = {}

    # Recursive function to assign codes
    def gen(node, code=""):
        if node:
            if node.sym is not None:  # leaf node
                codes[node.sym] = code
            gen(node.left, code + "0")
            gen(node.right, code + "1")

    gen(root)
    return codes


# Build Huffman codes
hf_codes = huffman(freq)

# Encode the image using Huffman codes
hf_bits_list = [hf_codes[p] for p in flat]
hf_encoded = ''.join(hf_bits_list)

# Save Huffman encoded data (⚠️ very large text file)
with open("outputs/task3/huffman.txt", "w") as f:
    f.write(hf_encoded)


# -----------------------------
# STEP 4: Compression Statistics
# -----------------------------

# Original size (each pixel = 8 bits)
original_bits = len(flat) * 8

# Encoded sizes
sf_bits = len(sf_encoded)
hf_bits = len(hf_encoded)

# Print results
print("Encoding done! Check outputs/task3/\n")

print("Original bits:", original_bits)
print("Shannon-Fano bits:", sf_bits)
print("Huffman bits:", hf_bits)

# Compression ratios (original / compressed)
print("Shannon-Fano Compression Ratio:", round(original_bits / sf_bits, 3))
print("Huffman Compression Ratio:", round(original_bits / hf_bits, 3))
