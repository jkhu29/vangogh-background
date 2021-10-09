import torch
import cv2
import convert
import segment


p2v = convert.model().cuda()
p2v_state_dict = torch.load("convert/photo2vangogh.pth")
p2v.load_state_dict(p2v_state_dict)

seg = segment.model().cuda()
seg_state_dict = torch.load("segment/u2net.pth")
seg.load_state_dict(seg_state_dict)

img = cv2.imread("test.jpg")
img_torch = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
img_seg_torch, _, _, _, _, _, _ = seg(img_torch)
img_seg = img_seg_torch.cpu().numpy().transpose(1, 2, 0)
cv2.imshow("seg", img_seg / 255)
cv2.waitKey()
cv2.destroyAllWindows()
