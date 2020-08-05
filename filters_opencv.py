from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import imutils

img = cv2.imread('image.jpg',cv2.IMREAD_COLOR)  
img2 = cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE) 
img1 = cv2.imread('background.jpg',cv2.IMREAD_GRAYSCALE)

height, width = img2.shape[0:2]
scale = 30
w = int(width*scale/100)
h = int(height*scale/100)

rimg = cv2.resize(img,(w,h))
rimg1 = cv2.resize(img1,(w,h))
rimg2 = cv2.resize(img2,(w,h))

#Aplicando un suavizado a la imagen
rimg1 = cv2.GaussianBlur(rimg1,(5,5),0)
rimg2 = cv2.GaussianBlur(rimg2,(5,5),0)

#Filtros Sobel para la imagen 1
f1 = cv2.Sobel(rimg1,cv2.CV_64F,1,0,ksize=3)
f2 = cv2.Sobel(rimg1,cv2.CV_64F,0,1,ksize=3)
f_sobel1 = (f1*0.5) + (f2*0.5)

img1_back = cv2.convertScaleAbs(f_sobel1) 

#Filtros sobel para la imagen 2
f3 = cv2.Sobel(rimg2,cv2.CV_64F,1,0,ksize=3)
f4 = cv2.Sobel(rimg2,cv2.CV_64F,0,1,ksize=3)
f_sobel2 = (f3*0.5) + (f4*0.5)

img2_back = cv2.convertScaleAbs(f_sobel2)

#Operaciòn RESTA para detecciòn de movimiento NUMPY
f = f_sobel2 - f_sobel1
resta_numpy = cv2.convertScaleAbs(f)

#Operación RESTA con OpenCV
difference_sobel = cv2.absdiff(f_sobel2,f_sobel1)
frameDelta = cv2.convertScaleAbs(difference_sobel)

#print(cv2.contourArea(max_cnts))

#SUMA pixeles RESTA
suma_pix_numpy = resta_numpy.sum()
suma_pix_cv2 = frameDelta.sum()

print('resta numpy {}'.format(suma_pix_numpy))
print('resta cv2 {}'.format(suma_pix_cv2))

cv2.imshow('FrameDelta',frameDelta)
cv2.imshow('Sobel_back_img1',img1_back)
cv2.imshow('Sobel_back_img2',img2_back)
cv2.waitKey();cv2.destroyAllWindows()

#Graficando Histogramas
plt.figure(figsize=(8,16))
plt.subplot(131);
plt.hist(frameDelta.flatten(),255,[0,255],histtype='bar')
plt.title('Histograma frameDelta')
plt.subplot(132);plt.hist(img2_back.flatten(),255,[0,255],histtype='bar')
plt.title('Histograma img2_back')
plt.subplot(133);plt.hist(frameDelta.flatten(),255,[0,255],histtype='bar')
plt.title('Histograma Resta')
plt.show()


