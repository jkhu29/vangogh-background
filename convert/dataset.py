import os
import cv2
import numpy as np
import tfrecord
from tfrecord.torch.dataset import TFRecordDataset

vangogh_path = "vangogh2photo/trainA"
photo_path = "vangogh2photo/trainB"
size_image = 128

writer = tfrecord.TFRecordWriter("train.tfrecord")
stride_vangogh = 32
stride_photo = 150
for vangogh_name, photo_name in zip(os.listdir(vangogh_path), os.listdir(photo_path)):
    print(vangogh_name, photo_name)
    img_vangogh = cv2.imread(os.path.join(vangogh_path, vangogh_name))
    img_photo = cv2.imread(os.path.join(photo_path, photo_name))

    for x in np.arange(0, img_vangogh.shape[0] - size_image + 1, stride_vangogh):
        for y in np.arange(0, img_vangogh.shape[1] - size_image + 1, stride_vangogh):
            img_part = img_vangogh[int(x): int(x + size_image),
                                   int(y): int(y + size_image)]
            img_part = img_part.transpose(2, 0, 1)

            for x1 in np.arange(0, img_photo.shape[0] - size_image + 1, stride_photo):
                for y1 in np.arange(0, img_photo.shape[1] - size_image + 1, stride_photo):
                    img_photo_part = img_photo[int(x1): int(x1 + size_image),
                                               int(y1): int(y1 + size_image)]
                    img_photo_part = img_photo_part.transpose(2, 0, 1)
                    writer.write({
                        "vangogh": (img_part.tobytes(), "byte"),
                        "photo": (img_photo_part.tobytes(), "byte"),
                        "size": (size_image, "int")
                    })
writer.close()

cnt = 0
description = {
    "vangogh": "byte",
    "photo": "byte",
    "size": "int",
}
for record in TFRecordDataset("train.tfrecord", None, description):
    cnt += 1
print("length of train dataset: {}".format(cnt))
