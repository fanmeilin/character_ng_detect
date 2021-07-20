from lib.common_lib import DataManager
from lib.common_lib import AnnotationManager
from lib.to_file import Box2File
from tqdm import tqdm
import cv2 as cv
import re
'''
通过json中的info信息获取圆心坐标
输入：rec表示一张图片的json信息
输出：查找图片的圆心坐标：形式[x,y]
'''
def getCicleByInfo(rec):
    centerX = 0
    centerY = 0
    circles = rec['info']['circles']
    for item in circles:
        centerX += item[0]
        centerY += item[1]
    centerX /= len(circles)
    centerY /= len(circles)
    result = [centerX,centerY]
    return result,circles[2][2],circles[3][2]

data = DataManager.load('./assets/new_word.json')
image_data_root = 'F:/mountMl_fan/bearing/'
word_classifier = Box2File()

am = AnnotationManager(data.class_dict)
img_ng_list = []
img_ng_info = []
word_list = []
for i,rec in enumerate(tqdm(data)):
    image_path = image_data_root + rec['info']['image_path']
    img = cv.imread(image_path)
    center,r_inner,r_outer = getCicleByInfo(rec)
    rec_id = rec['info']['uuid']
    word_img = False
    bbox_list = []
    for inst in rec['instances']:
        character_id = inst['uuid']
        class_names = am.get_classname(inst)
        if('word' in class_names):
            word_img = True
            Info = am.get_xyxy(inst)
            bbox = [[Info[0],Info[1]],[Info[2],Info[3]]]
            bbox_list.append(bbox)
    if(word_img):
        word_list.append( rec['info']['image_path'])
        pattern_list = rec['info']['content']
        word_classifier.save_img(img,rec_id+"_"+character_id,bbox_list,r_inner,r_outer,center,pattern_list)