import torch
import cv2
from skimage.color import rgb2yiq, yiq2rgb

import convert
import segment
import utils


p2v = convert.model().cuda()
p2v_state_dict = torch.load("convert/pretrain.pth")
p2v.load_state_dict(p2v_state_dict)

img = cv2.imread("test.jpg")
# img = cv2.resize(img, (300, 240))
img_yiq = rgb2yiq(img)
img_torch = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
img_vangogh_torch = p2v(img_torch)
img_vangogh = img_vangogh_torch[0].detach().cpu().numpy().transpose(1, 2, 0)
g = utils.lum(img_vangogh.transpose(2, 0, 1))
img_yiq[..., 0] = g
img_vangogh = yiq2rgb(img_yiq)

cv2.imshow("seg", img_vangogh)
cv2.waitKey()
cv2.destroyAllWindows()
