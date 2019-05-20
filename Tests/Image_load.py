import cv2
img = cv2.imread("1.png")
print(img, '\n\n')

for i in img:
    for j in i:
        list = j.tolist()
        if int(list[0]) == 0:
            print(0, end=', ')
        else:
            print(1, end=', ')

