from lib.common_lib import DataManager
from lib.common_lib import AnnotationManager
from lib.core import Word_Classification
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
    result = [centerX, centerY]
    return result, circles[2][2], circles[3][2]
# 根据pattern——list判断识别情况


def find_ng_img(data_json, image_data_root, distribution_classes, weights_path, pattern_list, re_file):
    word_classifier = Word_Classification(weights_path, distribution_classes)
    am = AnnotationManager(data_json.class_dict)
    img_ng_list = []
    img_ng_info = []
    word_list = []
    result_list = []
    for rec in tqdm(data_json):
        image_path = image_data_root + rec['info']['image_path']
        if not(re.match(re_file, rec['info']['image_path'])):
            # print("pass")
            continue
        img = cv.imread(image_path)
        center, r_inner, r_outer = getCicleByInfo(rec)
        # rec_id = rec['info']['uuid']
        word_img = False
        bbox_list = []
        for inst in rec['instances']:
            # character_id = inst['uuid']
            class_names = am.get_classname(inst)
            if('word' in class_names):
                word_img = True
                Info = am.get_xyxy(inst)
                bbox = [[Info[0], Info[1]], [Info[2], Info[3]]]
                bbox_list.append(bbox)
        if(word_img):
            word_list.append(rec['info']['image_path'])  # 图片路径
            is_ng, result = word_classifier.get_str_matchInfo(
                img, bbox_list, r_inner, r_outer, center, pattern_list)
            result_list.append(result)
            if(is_ng):
                img_ng_list.append(img)
                img_ng_info.append([result, image_path])  # 图片路径的index 预测的图片信息
                # print(rec_id,image_path)
                # print(rec['info']['image_path'],result)
                # print()
    print("all pattern pred img:", len(word_list))
    print("all error pred img:", len(img_ng_info))
    print("error pred img---------------------------------------------------------------------")
    for info in img_ng_info:
        print(info[0], info[1])
# 使用content预测（正则）


def find_ng_img_bycontent(data_json, image_data_root, distribution_classes, weights_path):
    word_classifier = Word_Classification(weights_path, distribution_classes)
    am = AnnotationManager(data_json.class_dict)
    img_ng_list = []
    img_ng_info = []
    word_list = []
    result_list = []
    for rec in tqdm(data_json):
        image_path = image_data_root + rec['info']['image_path']
        img = cv.imread(image_path)
        center, r_inner, r_outer = getCicleByInfo(rec)
        # rec_id = rec['info']['uuid']
        word_img = False
        bbox_list = []
        for inst in rec['instances']:
            # character_id = inst['uuid']
            class_names = am.get_classname(inst)
            if('word' in class_names):
                word_img = True
                Info = am.get_xyxy(inst)
                bbox = [[Info[0], Info[1]], [Info[2], Info[3]]]
                bbox_list.append(bbox)
        if(word_img):
            word_list.append(rec['info']['image_path'])  # 图片路径
            pattern_list = rec["info"]["content"]
            is_ng, result = word_classifier.get_str_matchInfo(
                img, bbox_list, r_inner, r_outer, center, pattern_list)
            result_list.append(result)
            if(is_ng):
                img_ng_list.append(img)
                img_ng_info.append([result, image_path])  # 图片路径的index 预测的图片信息
                # print(rec_id,image_path)
                # print(rec['info']['image_path'],result)
                # print()
    print("all pattern pred img:", len(word_list))
    print("all error pred img:", len(img_ng_info))
    acc = (1-len(img_ng_info)/len(word_list))*100
    print(f"acc = {acc:.2f} %")
    print("error pred img---------------------------------------------------------------------")
    for info in img_ng_info:
        print(info[0], info[1])

# 使用content预测（非正则，仅仅忽略非字符和数字）


def find_ng_imgstr_bycontent(data_json, image_data_root, distribution_classes, weights_path):
    word_classifier = Word_Classification(weights_path, distribution_classes)
    am = AnnotationManager(data_json.class_dict)
    img_ng_list = []
    img_ng_info = []
    word_list = []
    result_list = []
    for rec in tqdm(data_json):
        image_path = image_data_root + rec['info']['image_path']
        img = cv.imread(image_path)
        center, r_inner, r_outer = getCicleByInfo(rec)
        # rec_id = rec['info']['uuid']
        word_img = False
        bbox_list = []
        for inst in rec['instances']:
            # character_id = inst['uuid']
            class_names = am.get_classname(inst)
            if('word' in class_names):
                word_img = True
                Info = am.get_xyxy(inst)
                bbox = [[Info[0], Info[1]], [Info[2], Info[3]]]
                bbox_list.append(bbox)
        if(word_img):
            word_list.append(rec['info']['image_path'])  # 图片路径
            pattern_list = rec["info"]["content"]
            is_ng, result = word_classifier.get_str_Info(
                img, bbox_list, r_inner, r_outer, center, pattern_list)
            result_list.append(result)
            if(is_ng):
                img_ng_list.append(img)
                img_ng_info.append([result, image_path])  # 图片路径的index 预测的图片信息
    print("all pattern pred img:", len(word_list))
    print("all error pred img:", len(img_ng_info))
    print("error pred img---------------------------------------------------------------------")
    for info in img_ng_info:
        print(info[0], info[1])

# 判断starving 文件的pattern类别


def find_starving_cluster(data_json, image_data_root, distribution_classes, weights_path, pattern_list, re_file):
    word_classifier = Word_Classification(weights_path, distribution_classes)
    am = AnnotationManager(data_json.class_dict)
    word_list = []
    result_list3 = []
    result_list2 = []
    result_list0 = []
    for rec in tqdm(data_json):
        image_path = image_data_root + rec['info']['image_path']
        if not(re.match(re_file, rec['info']['image_path'])):
            # print("pass")
            continue
        img = cv.imread(image_path)
        center, r_inner, r_outer = getCicleByInfo(rec)
        # rec_id = rec['info']['uuid']
        word_img = False
        bbox_list = []
        for inst in rec['instances']:
            # character_id = inst['uuid']
            class_names = am.get_classname(inst)
            if('word' in class_names):
                word_img = True
                Info = am.get_xyxy(inst)
                bbox = [[Info[0], Info[1]], [Info[2], Info[3]]]
                bbox_list.append(bbox)
        if(word_img):
            word_list.append(rec['info']['image_path'])  # 图片路径
            _, result = word_classifier.get_str_matchInfo(
                img, bbox_list, r_inner, r_outer, center, pattern_list)
            if(len(result["str_list"]) == 3):
                result_list3.append([result, image_path])
            elif(len(result["str_list"]) == 2):
                result_list2.append([result, image_path])
            else:
                result_list0.append([result, image_path])
            # result_list.append(result)
    print("all pattern pred img:", len(word_list))
    # print("all error pred img:",len(img_ng_info))
    print("pred img size 3--------------------------------------------------------------------------")
    for info in result_list3:
        print(info[0], info[1])
    print("pred img size 2---------------------------------------------------------------------------")
    for info in result_list2:
        print(info[0], info[1])
    print("pred img size another---------------------------------------------------------------------")
    for info in result_list0:
        print(info[0], info[1])

# 保存更新的json  只包含word img 添加content


def revise_json(data_json, image_data_root, distribution_classes, weights_path, pattern_list):
    word_classifier = Word_Classification(weights_path, distribution_classes)
    am = AnnotationManager(data_json.class_dict)
    word_list = []
    # record = {"record":[]}
    for rec in tqdm(data_json):
        image_path = image_data_root + rec['info']['image_path']
        parent_file = image_path.split('/')[-3]
        is_ok = True
        if(parent_file != "ok"):
            is_ok = False
        if(is_ok):
            rec['info']['content'] = ["6202/P6", "BH"]
            continue
        img = cv.imread(image_path)
        center, r_inner, r_outer = getCicleByInfo(rec)
        # rec_id = rec['info']['uuid']
        word_img = False
        bbox_list = []
        for inst in rec['instances']:
            # character_id = inst['uuid']
            class_names = am.get_classname(inst)
            if('word' in class_names):
                word_img = True
                Info = am.get_xyxy(inst)
                bbox = [[Info[0], Info[1]], [Info[2], Info[3]]]
                bbox_list.append(bbox)
        if(word_img):
            # if(is_ok):
            #     # record["record"].append(rec)
            #     continue
            word_list.append(rec['info']['image_path'])  # 图片路径
            _, result = word_classifier.get_str_matchInfo(
                img, bbox_list, r_inner, r_outer, center, pattern_list)
            if(len(result["str_list"]) == 3):
                rec['info']['content'] = ["6202/P6YB", "1621F1", "BH"]
            elif(len(result["str_list"]) == 2):
                rec['info']['content'] = ["6202SPL", "BH"]
            # record["record"].append(rec)
    return data_json


# ### 生成添加content的json文件
# data_json = DataManager.load('./assets/word.json')
# image_data_root = 'F:/mountMl_fan/bearing/'
# distribution_classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# weights_path = "./assets/resnet34.pt"
# pattern_list = ["6202/P6","BH"]
# data_json = revise_json(data_json,image_data_root,distribution_classes,weights_path,pattern_list)
# data_json.save_json("assets/new_word.json")

# 根据content 检测错误情况
data_json = DataManager.load('./assets/new_word.json')
image_data_root = 'F:/mountMl_fan/bearing/'
distribution_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                        'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
weights_path = "./assets/levit_128s.pt"

# "data/2021-05-26/202/starving/"
find_ng_img_bycontent(data_json, image_data_root,
                      distribution_classes, weights_path)
