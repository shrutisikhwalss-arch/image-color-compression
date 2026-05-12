import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def load_image(source):
    """
    Load an image from a file path or uploaded file object.
    Always returns a uint8 RGB numpy array.
    """
    img = Image.open(source).convert("RGB")
    return np.array(img)


def compress_with_kmeans(image_array, n_colors=16):
    """
    Compress image using KMeans color quantization.
    Reduces the image to n_colors unique colors.
    Returns the recolored image as uint8 array.
    """
    data = image_array / 255.0
    h, w, _ = image_array.shape
    data_flat = data.reshape(-1, 3)

    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(data_flat)

    new_colors = kmeans.cluster_centers_[kmeans.predict(data_flat)]
    recolored = (new_colors.reshape(h, w, 3) * 255).astype(np.uint8)

    return recolored


def get_image_size_kb(image_array, format="PNG"):
    """
    Returns the in-memory file size of an image array in KB.
    Simulates saving to disk without actually writing a file.
    """
    import io
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.tell() / 1024


def get_size_reduction(original_array, compressed_array, format="PNG"):
    """
    Returns original size, compressed size, and reduction percentage.
    """
    original_kb = get_image_size_kb(original_array, format)
    compressed_kb = get_image_size_kb(compressed_array, format)
    reduction = (1 - compressed_kb / original_kb) * 100
    return original_kb, compressed_kb, reduction