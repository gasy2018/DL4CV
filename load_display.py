import cv2
image = cv2.imread("chap1/images/massabe.jpg")
print(image.shape)

(b_1, g_1, r_1) = image[20, 100] # accesses pixel at x=100, y=20
(b_2, g_2, r_2) = image[75, 25] # accesses pixel at x=25, y=75
(b_3, g_3, r_3) = image[90, 85] # accesses pixel at x=85, y=90



cv2.imshow("Image", image)
cv2.waitKey(0)

