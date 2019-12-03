import os
import cv2

src_dir = '/home/iim/Data/deepglint_emotion'

count = 0
for root, dirs, files in os.walk(src_dir):
	for name in files:
		count += 1
		if count % 1000 == 0:
			print(count)
		img_path = os.path.join(root, name)
		img = cv2.imread(img_path)
		img = cv2.resize(img, (112,112))
		cv2.imwrite(img_path, img)
