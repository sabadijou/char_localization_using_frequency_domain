import cv2
import numpy as np

full_text = cv2.imread(r'input/text.png', cv2.IMREAD_GRAYSCALE)
selcted_char = cv2.imread(r'input/e.png', cv2.IMREAD_GRAYSCALE)

########################################################

nRows = 32
mCols = 32
img = full_text
sizeX = img.shape[1]
sizeY = img.shape[0]
text_slices = []
for i in range(0,nRows):
    for j in range(0, mCols):
        roi = img[int(i*sizeY/nRows):int(i*sizeY/nRows + sizeY/nRows), int(j*sizeX/mCols):int(j*sizeX/mCols + sizeX/mCols)]
        text_slices.append(roi)

######################################################

selcted_char_float = np.float32(selcted_char)
h = cv2.dft(np.float32(selcted_char_float),flags = cv2.DFT_COMPLEX_OUTPUT)
h = dft_shift = np.fft.fftshift(h)
convalved_parts = []
for slice in text_slices:
    full_text_float = np.float32(slice)
    i = cv2.dft(full_text_float,flags = cv2.DFT_COMPLEX_OUTPUT)
    i = dft_shift = np.fft.fftshift(i)
    i = h*i
    max_i = np.max(i)
    th = 0.85 * max_i
    i = i * th
    i = np.fft.ifftshift(i)
    i = cv2.idft(i, flags=cv2.DFT_SCALE)
    i = cv2.magnitude(i[:, :, 0], i[:, :, 1])

    convalved_parts.append(i)
def concat_vh(list_2d):
    return cv2.hconcat([cv2.vconcat(list_h)
                        for list_h in list_2d])
k_list = []
i = 0

while i < nRows * nRows:
    k_list.append(cv2.hconcat(convalved_parts[i:i+nRows]))
    i += nRows
final_image = cv2.vconcat(k_list[0:nRows])
cv2.imshow(' ', final_image)
cv2.imwrite(r'results/result.jpg',final_image)
cv2.waitKey()

##################################################################
