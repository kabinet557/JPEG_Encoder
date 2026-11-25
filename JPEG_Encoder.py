import numpy as np
from PIL import Image
from scipy.fft import dctn
import struct
import heapq
from collections import Counter

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

def generate_zigzag_indices(N):
    indices = []
    for diagonal in range(2*N - 1):
        start = max(0, diagonal - (N - 1))
        end = min(N - 1, diagonal)

        if diagonal % 2 == 0:
            rows = range(end, start - 1, -1)
        else:
            rows = range(start, end + 1)

        for row in rows:
            col = diagonal - row
            indices.append((row, col))
    return indices


def zigzag_scan(block):
    N = block.shape[0]
    indices = generate_zigzag_indices(N)
    return np.array([
        block[row, col]
        for row, col in indices
    ])
Y_zigzag_all = np.array([
    [zigzag_scan(block) for block in row]
    for row in Y_quantized
])

Cb_zigzag_all = np.array([
    [zigzag_scan(block) for block in row]
    for row in Cb_quantized
])

Cr_zigzag_all = np.array([
    [zigzag_scan(block) for block in row]
    for row in Cr_quantized
])
print("Y zigzag shape:", Y_zigzag_all.shape)
print("Cb zigzag shape:", Cb_zigzag_all.shape)
print("Cr zigzag shape:", Cr_zigzag_all.shape)

print("\nFirst Y block after zigzag:")
print(Y_zigzag_all[0, 0])

print("\nFirst Cb block after zigzag:")
print(Cb_zigzag_all[0, 0])

print("\nFirst Cr block after zigzag:")
print(Cr_zigzag_all[0, 0])

def get_category(value):
    if value == 0:
        return 0
    return abs(value).bit_length()


def get_amplitude_bits(value):
    if value == 0:
        return ""
    size = get_category(value)
    if value > 0:
        amplitude = value
    else:
        amplitude = (1 << size) - 1 + value
    return format(amplitude, f'0{size}b')

def encode_dc(current_dc, previous_dc):
    difference = int(current_dc) - int(previous_dc)
    category = get_category(difference)
    amplitude = get_amplitude_bits(difference)
    return difference, category, amplitude

def encode_ac(ac_coefficients):
    symbols = []
    zero_run = 0

    for coefficient in ac_coefficients:
        coefficient = int(coefficient)
        if coefficient == 0:
            zero_run += 1
        else:
            # Flush pending runs of 16 zeros ONLY because a non-zero value follows
            while zero_run >= 16:
                symbols.append((15, 0))
                zero_run -= 16
            size = get_category(coefficient)
            symbols.append((zero_run, size, coefficient))
            zero_run = 0
    # If there were any trailing zeros at the end of the block, emit a single EOB
    if zero_run > 0:
        symbols.append((0, 0))
    return symbols

BITS_DC_LUM = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
VALS_DC_LUM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

BITS_DC_CHR = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
VALS_DC_CHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

BITS_AC_LUM = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
VALS_AC_LUM = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
    0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
    0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
    0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
]

BITS_AC_CHR = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
VALS_AC_CHR = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41,
    0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xD1, 0x15, 0x62, 0x72, 0xF0,
    0x24, 0x34, 0x43, 0x82, 0x92, 0x53, 0xE1, 0x0A, 0x16, 0x17, 0x18, 0x19,
    0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
    0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2,
    0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5,
    0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8,
    0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
]


def expand_huffman_lut(bits, vals):
    """Expands Annex K (BITS, VALS) into a direct {symbol: bitstring}LUT."""
    lut = {}
    code = 0
    idx = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            lut[vals[idx]] = format(code, f"0{length}b")
            code += 1
            idx += 1
        code <<= 1
    return lut


DC_LUM_LUT = expand_huffman_lut(BITS_DC_LUM, VALS_DC_LUM)
DC_CHR_LUT = expand_huffman_lut(BITS_DC_CHR, VALS_DC_CHR)
AC_LUM_LUT = expand_huffman_lut(BITS_AC_LUM, VALS_AC_LUM)
AC_CHR_LUT = expand_huffman_lut(BITS_AC_CHR, VALS_AC_CHR)


def encode_block(zz_block, prev_dc, dc_lut, ac_lut):
    dc_val = int(zz_block[0])
    _, dc_cat, dc_amp = encode_dc(dc_val, prev_dc)
    bits = [dc_lut[dc_cat], dc_amp]

    for sym in encode_ac(zz_block[1:]):
        if len(sym) == 2:  # EOB (0, 0) or ZRL (15, 0)
            run, size = sym
            bits.append(ac_lut[(run << 4) | size])
        else:              # (zero_run, size, coeff)
            run, size, coeff = sym
            bits.append(ac_lut[(run << 4) | size])
            bits.append(get_amplitude_bits(coeff))

    return "".join(bits), dc_val


# Ensure Y block grid has exact 2x2 coverage per Cb/Cr block for 4:2:0 MCUs
cb_rows, cb_cols, _ = Cb_zigzag_all.shape
pad_y_r = max(0, 2 * cb_rows - Y_zigzag_all.shape[0])
pad_y_c = max(0, 2 * cb_cols - Y_zigzag_all.shape[1])
if pad_y_r > 0 or pad_y_c > 0:
    Y_zigzag_all = np.pad(Y_zigzag_all, ((0, pad_y_r), (0, pad_y_c), (0, 0)), mode="edge")

prev_dc_y = prev_dc_cb = prev_dc_cr = 0
bit_chunks = []

for r in range(cb_rows):
    for c in range(cb_cols):
        # 4 Luminance blocks (top-left, top-right, bottom-left, bottom-right)
        for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
            blk_bits, prev_dc_y = encode_block(
                Y_zigzag_all[2 * r + dr, 2 * c + dc], prev_dc_y, DC_LUM_LUT, AC_LUM_LUT
            )
            bit_chunks.append(blk_bits)

        # 1 Cb block + 1 Cr block
        cb_bits, prev_dc_cb = encode_block(Cb_zigzag_all[r, c], prev_dc_cb, DC_CHR_LUT, AC_CHR_LUT)
        cr_bits, prev_dc_cr = encode_block(Cr_zigzag_all[r, c], prev_dc_cr, DC_CHR_LUT, AC_CHR_LUT)
        bit_chunks.append(cb_bits)
        bit_chunks.append(cr_bits)

bitstring = "".join(bit_chunks)

# Pad final byte with 1s and apply mandatory 0xFF -> 0xFF 0x00 byte stuffing
pad_len = (8 - len(bitstring) % 8) % 8
bitstring += "1" * pad_len
raw_entropy = int(bitstring, 2).to_bytes(len(bitstring) // 8, "big")
stuffed_entropy = raw_entropy.replace(b"\xff", b"\xff\x00")

def make_dqt(table_id, q_matrix):
    zz_q = zigzag_scan(q_matrix).astype(np.uint8).tobytes()
    return b"\xff\xdb" + struct.pack(">HB", 67, table_id) + zz_q


def make_dht(table_class, table_id, bits, vals):
    info = (table_class << 4) | table_id
    payload = bytes([info]) + bytes(bits) + bytes(vals)
    return b"\xff\xc4" + struct.pack(">H", len(payload) + 2) + payload


H, W = Y_raw.shape
soi = b"\xff\xd8"
app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
dqt = make_dqt(0, QY) + make_dqt(1, QC)
sof0 = (
    b"\xff\xc0"
    + struct.pack(">HBHHB", 17, 8, H, W, 3)
    + bytes([
        1, 0x22, 0,  # Y:  2x2 sampling, quant table 0
        2, 0x11, 1,  # Cb: 1x1 sampling, quant table 1
        3, 0x11, 1,  # Cr: 1x1 sampling, quant table 1
    ])
)
dht = (
    make_dht(0, 0, BITS_DC_LUM, VALS_DC_LUM)
    + make_dht(1, 0, BITS_AC_LUM, VALS_AC_LUM)
    + make_dht(0, 1, BITS_DC_CHR, VALS_DC_CHR)
    + make_dht(1, 1, BITS_AC_CHR, VALS_AC_CHR)
)
sos = (
    b"\xff\xda"
    + struct.pack(">HB", 12, 3)
    + bytes([
        1, 0x00,  # Y:  DC table 0, AC table 0
        2, 0x11,  # Cb: DC table 1, AC table 1
        3, 0x11,  # Cr: DC table 1, AC table 1
        0x00, 0x3F, 0x00,
    ])
)
eoi = b"\xff\xd9"

output_path = "C:\\Users\\abhin\\Documents\\PYNQ_Projects\\JPEG_Encoder\\Outputs\\big_building.jpg"
with open(output_path, "wb") as f:
    f.write(soi + app0 + dqt + sof0 + dht + sos + stuffed_entropy + eoi)

print("Saved JPEG successfully to:", output_path)