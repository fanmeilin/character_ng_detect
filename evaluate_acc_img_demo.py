from torch.utils.data import DataLoader, Dataset
import os,glob
from lib.evaluate import Binary_str
import cv2 as cv

distribution_classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# weights_path = "./assets/resnet34.pt"
weights_path_levit = "./assets/levit_128s.pt"
model_name = "levit_128s"
cc_obj = Binary_str(model_name,weights_path_levit,distribution_classes)
img_root = "F:/localTest/img_cluster/"
all_pred_right = 0
all_sample = 0
for label in os.listdir(img_root):
    imt_group = img_root+label   
    group_path = [path for path in glob.glob(imt_group+"/*.png")]
    group = [cv.resize(cv.imread(path,0), (224,224))for path in group_path]
    test_loader = DataLoader(dataset = group, batch_size=16, shuffle=False)
    right_total = 0
    for test in test_loader:
        result_pred = cc_obj.get_pred_str(test)
        right = result_pred.count(label)
        right_total += right
#         print(result_pred)
#         break
    label_acc = right_total/len(group)
    print(f"{label} acc:",label_acc)
    all_sample += len(group)
    all_pred_right += right_total
print("all_acc:",all_pred_right/all_sample)