import numpy as np
import gzip

def extract_images(filename, num_images, size_img):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(size_img * size_img * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, size_img * size_img)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels

def convLayer_weights(size):
    standard_dev = 1/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0.0, scale = standard_dev, size = size)

def fullyConnected_weights(size):
    return np.random.standard_normal(size = size)*0.01




