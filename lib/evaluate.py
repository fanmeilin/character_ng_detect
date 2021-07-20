import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
import cv2

class ViTBase16(nn.Module):
    def __init__(self,model_name,n_classes,pretrained = False):
        super(ViTBase16,self).__init__()
        self.model = timm.create_model(model_name,pretrained=pretrained,in_chans =1,num_classes=n_classes)
    def forward(self,x):
        x = self.model(x)
        x = x.argmax(dim=1)
        return x

class Binary_str:
    def __init__(self,model_name,weights_path,distribution_classes,pretrained = False):
        n_classes = len(distribution_classes)
        self.model = ViTBase16(model_name,n_classes=n_classes,pretrained=pretrained) #模型为实例化 ViTBase16
        self.model.load_state_dict(torch.load(weights_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  #!
#         if self.device.type=="cuda":
#             self.model.to(device)
        self.distribution_classes = distribution_classes
    @staticmethod
    def process(group):
        img_list = []
        # resize = transforms.Resize([224,224])
        toTensor = transforms.ToTensor()
        for img in group:
            img = img.numpy()
            if img.ndim!=2:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转化为灰度图
            #二值化
#             blur = cv.GaussianBlur(img,(5,5),0)
#             ret,thImg = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)  
#             mask = cv.erode(thImg.astype('uint8'), kernel=np.ones((3,3)))
            mask = img
            #resize and normalize
            mask = Image.fromarray(mask)
            mask = toTensor(mask)
            img_list.append(mask.numpy())
        return torch.tensor(img_list) 
    
    def get_pred_str(self,group):
        img_list = Binary_str.process(group)
        if self.device.type=="cuda":
            img_list = img_list.cuda()
        output = self.model(img_list)
        result = ""
        for index in output:
            result += self.distribution_classes[index] 
        return result