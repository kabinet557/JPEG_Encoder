import numpy as np
from PIL import Image
from scipy.fft import dctn

raw_image = Image.open("C:\\Users\\abhin\\Documents\\PYNQ_Projects\\JPEG_Encoder\\rgb8bit\\big_building.ppm")

rgb_raw = np.array(raw_image, dtype=np.float64)
R_raw = rgb_raw[:, :, 0]
G_raw = rgb_raw[:, :, 1]
B_raw = rgb_raw[:, :, 2]

Y_raw = 0.299 * R_raw + 0.587 * G_raw + 0.114 * B_raw
Cb_raw = 128 - 0.168736 * R_raw - 0.331264 * G_raw + 0.5 * B_raw
Cr_raw = 128 + 0.5 * R_raw - 0.418688* G_raw - 0.081312* B_raw

Y_raw = np.clip(Y_raw, 0, 255).astype(np.uint8)
Cb_raw = np.clip(Cb_raw, 0, 255).astype(np.uint8)
Cr_raw = np.clip(Cr_raw, 0, 255).astype(np.uint8)

Image.fromarray(Y_raw).save("C:\\Users\\abhin\\Documents\\PYNQ_Projects\\JPEG_Encoder\\Outputs\\Y.png")
Image.fromarray(Cb_raw).save("C:\\Users\\abhin\\Documents\\PYNQ_Projects\\JPEG_Encoder\\Outputs\\Cb.png")
Image.fromarray(Cr_raw).save("C:\\Users\\abhin\\Documents\\PYNQ_Projects\\JPEG_Encoder\\Outputs\\Cr.png")

def chroma_subsample(channel):

    H, W = channel.shape
    if H % 2 != 0 or W % 2 != 0:
        channel = np.pad(
            channel,
            ((0, H % 2), (0, W % 2)),
            mode='edge'
        )
    subsampled = channel.reshape(
        channel.shape[0] // 2, 2,
        channel.shape[1] // 2, 2
    ).mean(axis=(1, 3))

    return np.round(subsampled).astype(np.uint8)


Cb_subsampled = chroma_subsample(Cb_raw)
Cr_subsampled = chroma_subsample(Cr_raw)

print("Y:", Y_raw.shape)
print("Cb:", Cb_subsampled.shape)
print("Cr:", Cr_subsampled.shape)

# used int16 because of negative values after shifting
Y_shifted = Y_raw.astype(np.int16) - 128
Cb_shifted = Cb_subsampled.astype(np.int16) - 128
Cr_shifted = Cr_subsampled.astype(np.int16) - 128

#just verifying the range of values after shifting
print("Y:", Y_shifted.min(), Y_shifted.max())
print("Cb:", Cb_shifted.min(), Cb_shifted.max())
print("Cr:", Cr_shifted.min(), Cr_shifted.max())

def form_blocks(channel):
    H, W = channel.shape
    pad_H = (8 - H % 8) % 8
    pad_W = (8 - W % 8) % 8
    channel = np.pad(
        channel,
        ((0, pad_H), (0, pad_W)),
        mode='edge'
    )
    H, W = channel.shape
    blocks = channel.reshape(
        H // 8, 8,
        W // 8, 8
    ).transpose(0, 2, 1, 3)
    return blocks

Y_blocks = form_blocks(Y_shifted)
Cb_blocks = form_blocks(Cb_shifted)
Cr_blocks = form_blocks(Cr_shifted)

print("Y blocks:", Y_blocks.shape)
print("Cb blocks:", Cb_blocks.shape)
print("Cr blocks:", Cr_blocks.shape)

print("First Y block:")
print(Y_blocks[0, 0])

#applying dct to each 8x8 block of Y, Cb, and Cr channels

Y_dct = dctn(Y_blocks, type=2, norm='ortho', axes=(-2, -1))
Cb_dct = dctn(Cb_blocks, type=2, norm='ortho', axes=(-2, -1))
Cr_dct = dctn(Cr_blocks, type=2, norm='ortho', axes=(-2, -1))

#luminance quantization matrix
QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Y_quantized = np.round(Y_dct / QY).astype(np.int16)

QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

Cb_quantized = np.round(Cb_dct / QC).astype(np.int16)
Cr_quantized = np.round(Cr_dct / QC).astype(np.int16)

print("DCT coefficients:")
print(Y_dct[0, 0])

print("\nQuantized coefficients:")
print(Y_quantized[0, 0])
print(Cb_quantized[0, 0])
print(Cr_quantized[0, 0])