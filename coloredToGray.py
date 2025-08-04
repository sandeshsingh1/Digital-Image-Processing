import cv2
import numpy as np
img=cv2.imread("lena.png")
gray_img=[]
for row in img:
    gray_row=[]
    for pixel in row:
        b=pixel[0]
        g=pixel[1]
        r=pixel[2]
        gray=int(0.299*r + 0.587*g + 0.114*b)
        gray=min(max(gray,0),255)
        gray_row.append(gray)
    gray_img.append(gray_row)
gray_img=np.array(gray_img,dtype=np.uint8)
bw_img=[]
for row1 in gray_img:
    bw_row=[]
    for pixel1 in row1:
        if(pixel1>127):
            bw_row.append(255)
        else:
            bw_row.append(0)
    bw_img.append(bw_row)
bw_img1=np.array(bw_img,dtype=np.uint8)
blimg=img.copy()
blimg[:,:,0]=0
blimg[:,:,1]=0
while True:
    cv2.imshow("Red-Image",blimg)
    cv2.imshow("Coloured-Image",img)
    cv2.imshow("Gray-Image",gray_img)
    cv2.imshow("Black-white",bw_img1)
    if cv2.waitKey(1) & 0xFF==ord("x"):
        break
cv2.destroyAllWindows()