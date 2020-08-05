from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

img1 = cv2.imread('image1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg',cv2.IMREAD_GRAYSCALE)

height,width = img1.shape[0:2]
scale = 30
w = int((width*scale)/100)
h = int((height*scale)/100)

rimg1 = cv2.resize(img1,(w,h))
rimg2 = cv2.resize(img2,(w,h))

#Calculando la TRANSFORMADA DE FOURIER
f1 = np.fft.fft2(rimg1)
f2 = np.fft.fft2(rimg2)

#corremos la componente de frecuencia de cero al centro
fshift1 = np.fft.fftshift(f1)
fshift2 = np.fft.fftshift(f2)

#obtenemos la magnitud del espectro
mg_spectrum1 = 20 * np.log(np.abs(fshift1))
mg_spectrum2 = 20 * np.log(np.abs(fshift2))

#Graficas de la magnitud del espectro para cada imagem
plt.figure(figsize=(8,8))
#Imagen 1
plt.subplot(221);plt.imshow(rimg1, cmap = plt.cm.gray)
plt.title('Imagen de entrada 1');plt.axis('off')
plt.subplot(222);plt.imshow(mg_spectrum1, cmap= plt.cm.Reds)
plt.title('Magnitude Spectrum 1');plt.axis('off')
#Imagen 2
plt.subplot(223);plt.imshow(rimg2, cmap = plt.cm.gray)
plt.title('Imagen de entrada 2');plt.axis('off')
plt.subplot(224);plt.imshow(mg_spectrum2, cmap= plt.cm.Reds)
plt.title('Magnitude Spectrum 2');plt.axis('off')
#Mostrar gràficos
#plt.show()

#Aplicando el FILTRO PASA ALTO
crows, ccolums = h//2, w//2
window_size = 15 
#Imagen 1
fshift1[crows-window_size : crows+window_size,
        ccolums-window_size : ccolums+window_size] = 0
mg_spectrum1_2 = 20 * np.log(np.abs(fshift1))

#Imagen 2
fshift2[crows-window_size : crows+window_size,
        ccolums-window_size : ccolums+window_size] = 0
mg_spectrum2_2 = 20 * np.log(np.abs(fshift2))

#Calculando la TRANSFORMADA INVERSA DE FOURIER

#Imagen 1
ifshift1 = np.fft.ifftshift(fshift1)
rimg1_back = np.abs(np.fft.ifft2(ifshift1))
plt.figure(figsize=(18,10))
plt.subplot(231);plt.imshow(rimg1, cmap=plt.cm.gray)
plt.subplot(232);plt.imshow(mg_spectrum1_2, cmap=plt.cm.Reds)
plt.subplot(233);plt.imshow(rimg1_back, cmap=plt.cm.gray)
plt.title('Imagen 1 filtrada');plt.axis('off')

#Imagen 2
ifshift2 = np.fft.ifftshift(fshift2)
rimg2_back = np.abs(np.fft.ifft2(ifshift2))
plt.subplot(234);plt.imshow(rimg2, cmap = plt.cm.gray)
plt.subplot(235);plt.imshow(mg_spectrum2_2, cmap = plt.cm.Reds)
plt.subplot(236);plt.imshow(rimg2_back, cmap = plt.cm.gray)
plt.title('Imagen 2 filtrada')
plt.show()

img_diff = cv2.absdiff(rimg1_back,rimg2_back)
plt.figure(figsize=(10,10))
plt.imshow(img_diff,cmap = plt.cm.gray)
plt.show()

pixelesnozero = cv2.sumElems(img_diff)
sumapixeles = img_diff.sum()
print('Suma de pixeles con numpy es: {0}'.format(sumapixeles))
print('Suma de piexeles con OpenCV es: {0}'.format(pixelesnozero))

cv2.imshow('image 1',rimg1)
cv2.imshow('image 2',rimg2)
cv2.waitKey();cv2.destroyAllWindows()
