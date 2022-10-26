import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import csv

std_thresh = 18
area_thresh = 600

def dist(m,a):
    return math.sqrt((abs(m[0]-a[0])**2) + (abs(m[1]-a[1]**2)))

def process(img, show = False) -> dict:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    ret,thresh = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
    edges = cv2.Canny(image=thresh, threshold1=100, threshold2=200) # Canny Edge Detection
    img_dilation = cv2.dilate(edges, None, iterations=4)
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros(np.shape(img))

    for i in range(len(contours)):
        cv2.drawContours(contour_img,contours,i,(255,0,0),3)

    defects = []
    img_area = np.shape(img_dilation)[0] * np.shape(img_dilation)[1]
    median = []
    mean_d = []
    std_d = []
    contour_thresh = []
    bh_count=0
    lof_count=0
    bh_area_percent=0
    lof_area_percent=0

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > area_thresh): 
            m = ((np.sum(contour, axis=0))/len(contour)).squeeze()
            median.append(m)
            contour_thresh.append(contour)
            cnt = contour.squeeze()
            d = []
            for a in cnt:
                d.append(dist(m, a))
            mean_d.append(np.mean(d))
            std_d.append(np.std(d))
            
            area_percent = (area / img_area) * 100

            isCircle = np.std(d) < std_thresh
            if isCircle:
                bh_count+=1
                bh_area_percent+=area_percent
            else:
                lof_count+=1
                lof_area_percent+=area_percent

            defects.append({
                'contour':contour,
                'median':m,
                'area':area,
                'area_percent':area_percent,
                'mean':np.mean(d),
                'std':np.std(d),
                'isBlowhole':isCircle
            })

    fig = plt.figure()

    if show:
        
        plt.imshow(contour_img, cmap='gray')
        for a in defects:
            if a['isBlowhole']:
                color='green'
                plt.plot(a['median'][0], a['median'][1], '.', color=color, label='Blowhole')
            else:
                color = 'red'
                plt.plot(a['median'][0], a['median'][1], '.', color=color, label='Lack of Fusion')
        plt.title('Defects')

    def_count = lof_count+bh_count
    defect_stats = {
        "def_count" : def_count,
        "lof_count" : lof_count,
        "lof_area_percent" : lof_area_percent,
        "bh_count" : bh_count,
        "bh_area_percent" : bh_area_percent,
    }

    return defect_stats, fig

def process_all_images(path):
    conditions = os.listdir(path)
    stats = {}
    for condition in conditions:
        stats[condition] = {}
        condition_path = path + "\\" + condition
        images = os.listdir(condition_path)
        for img_name in images:
            if "tif" in img_name or "jpg" in img_name:
                img = cv2.imread(condition_path+"\\"+img_name)
                stats[condition][img_name],fig = process(img, show=True)
    return stats, fig


def sum_stats(stats):
    stats_sum = []
    for condition in stats:
        condition_sum = {
            "condition" : condition,
            "def_count" : 0,
            "lof_count" : 0,
            "bh_count" : 0,
            "lof_area_percent" : 0,
            "bh_area_percent" : 0,
        }
        for stat in stats[condition]:
            for param in stats[condition][stat]:
                condition_sum[param] += stats[condition][stat][param]

        condition_sum["lof_area_percent"] = condition_sum["lof_area_percent"]/len(stats[condition])
        condition_sum["bh_area_percent"] = condition_sum["bh_area_percent"]/len(stats[condition])
        stats_sum.append(condition_sum)
    return stats_sum




def write_to_csv(list_of_dicts):
    with open("Data/output.csv", 'w') as f:
        field_names = list(list_of_dicts[0].keys())
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for dictionary in list_of_dicts:
            writer.writerow(dictionary)

if '__name__' == '__main__':
    path = r"Data\H13"
    stats = process_all_images(path)
    complete_stats = sum_stats(stats)
    write_to_csv(complete_stats)