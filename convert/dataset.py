import os
import cv2
import tfrecord
from tfrecord.torch.dataset import TFRecordDataset

vangogh_path = "vangogh2photo/trainA"
photo_path = "vangogh2photo/trainB"
size_image = 256

if not os.path.exists("train.tfrecord"):
    writer = tfrecord.TFRecordWriter("train.tfrecord")
    for _ in range(10):
        for vangogh_name, photo_name in zip(os.listdir(vangogh_path), os.listdir(photo_path)):
            print(vangogh_name, photo_name)
            img_vangogh = cv2.imread(os.path.join(vangogh_path, vangogh_name)).transpose(2, 0, 1)
            img_photo = cv2.imread(os.path.join(photo_path, photo_name)).transpose(2, 0, 1)
            _, size_image, _ = img_photo.shape
            writer.write({
                "vangogh": (img_vangogh.tobytes(), "byte"),
                "photo": (img_photo.tobytes(), "byte"),
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
    vangogh = record["vangogh"].reshape(
        3, size_image, size_image
    )
    vangogh_np = vangogh.transpose(1, 2, 0)
    cnt += 1
    if cnt < 1:
        cv2.imshow("res", vangogh_np)
        cv2.waitKey()
        cv2.destroyAllWindows()
print("length of train dataset: {}".format(cnt))
