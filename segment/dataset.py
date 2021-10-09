import os
import cv2
import numpy as np
import tfrecord

data_path = "dataset/training"

cnt = 0
size_image = 128
stride = 100
writer = tfrecord.TFRecordWriter("train.tfrecord")
for img_name in os.listdir(data_path):
    if "matte" in img_name:
        continue
    label_name = img_name.split(".")[0] + "_matte." + img_name.split(".")[1]
    print(img_name, label_name)
    cnt += 1

    img = cv2.imread(os.path.join(data_path, img_name))
    img = cv2.resize(img, (192 * 3, 192 * 4))
    img = img.transpose(2, 0, 1)

    label = cv2.imread(os.path.join(data_path, label_name), 0)
    label = cv2.resize(label, (192 * 3, 192 * 4))

    writer.write({
        "inputs": (img.tobytes(), "byte"),
        "labels": (label.tobytes(), "byte")
    })
writer.close()
print(cnt)
