from lib.common_lib import DataManager
from lib.common_lib import AnnotationManager
from lib.core import Word_Classification
from tqdm import tqdm
import cv2 as cv
import re
from multiprocessing import Process
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

def find_ng_img_bycontent(data_json, image_data_root, distribution_classes, weights_path,start_index,end_index):
    word_classifier = Word_Classification(weights_path, distribution_classes)
    am = AnnotationManager(data_json.class_dict)
    img_ng_list = []
    img_ng_info = []
    word_list = []
    result_list = []
    for rec in tqdm(data_json[start_index:end_index]):
        image_path = image_data_root + rec['info']['image_path']
        img = cv.imread(image_path)
        center, r_inner, r_outer = getCicleByInfo(rec)
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
                print(rec['info']['image_path'],result)
                print()
    print("all pattern pred img:", len(word_list))
    print("all error pred img:", len(img_ng_info))
    acc = (1-len(img_ng_info)/len(word_list))*100
    print(f"acc = {acc:.2f} %")
    print("error pred img---------------------------------------------------------------------")
    for info in img_ng_info:
        print(info[0], info[1])

# 使用content预测（非正则，仅仅忽略非字符和数字）


class MyProgress(Process):
    # 继承Progress类
    def __init__(self, index, data_json, image_data_root, distribution_classes, weights_path, start_index, end_index):
        super().__init__()
        self.index = index
        self.data_json = data_json
        self.image_data_root = image_data_root
        self.distribution_classes = distribution_classes
        self.weights_path = weights_path
        self.start_index = start_index
        self.end_index = end_index

    @staticmethod
    def find_ng_img_bycontent(data_json, image_data_root, distribution_classes, weights_path,start_index,end_index):
        word_classifier = Word_Classification(weights_path, distribution_classes)
        am = AnnotationManager(data_json.class_dict)
        img_ng_list = []
        img_ng_info = []
        word_list = []
        result_list = []
        for rec in tqdm(data_json[start_index:end_index]):
            image_path = image_data_root + rec['info']['image_path']
            img = cv.imread(image_path)
            center, r_inner, r_outer = getCicleByInfo(rec)
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
                    print(rec['info']['image_path'], result)
                    print()
        print("all pattern pred img:", len(word_list))
        print("all error pred img:", len(img_ng_info))
        acc = (1-len(img_ng_info)/len(word_list))*100
        print(f"acc = {acc:.2f} %")
        print("error pred img---------------------------------------------------------------------")
        for info in img_ng_info:
            print(info[0], info[1])

    def run(self):
        print('test %d多线程' % self.index)
        MyProgress.find_ng_img_bycontent(self.data_json, self.image_data_root, self.distribution_classes, self.weights_path, self.start_index, self.end_index)


if __name__ == '__main__':
    data_json = DataManager.load('./assets/new_word.json')
    image_data_root = 'F:/mountMl_fan/bearing/'
    distribution_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                            'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    weights_path = "./assets/levit_128s.pt"
    start_index = 0
    end_index = int(len(data_json)/2)
    process_list = []
    for i in range(2):
        p = MyProgress(i, data_json, image_data_root,
                        distribution_classes, weights_path,start_index,end_index)  # 定义p
        # p = Process(target=find_ng_img_bycontent, args=(data_json, image_data_root, distribution_classes, weights_path, start_index, end_index))
        
        process_list.append(p)
        start_index = end_index
        end_index = len(data_json)
    # print("start")
    for i in process_list:
        p.start()  # 进程准备就绪，等待cpu调度
        p.join()  # 阻塞上下文环境进程， 直到此方法进程终止
    print("end")
