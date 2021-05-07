

import math
import cv2
import numpy as np
import pandas as pd

from skimage import draw

# global variables
window_name = "Zadanie 1"
target_i = []
target_p = []

kernel_size = 1
sigma = 1

threshold1 = 0
threshold2 = 0

acc_res = 1
min_dist = 4
min_radius = 10
max_radius = 50

min_iou = 0.75
iris_seg_switch = 0



def load(image_path):
    loaded = cv2.imread(image_path)
    grey = cv2.merge([cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)]*3)
    return grey, loaded


def on_change_sigma(val, sb = False):
    global sigma
    sigma = val / 10

    if sb:
        process_sb()
    else:
        process()

def on_change_ks(val, tracker_name, sb = False):
    global kernel_size

    if val % 2 == 0:
        val += 1
        cv2.setTrackbarPos(tracker_name, window_name, val)

    kernel_size = val
    if sb:
        process_sb()
    else:
        process()
    
def on_change_th1(val):
    global threshold1
    threshold1 = val
    process()

def on_change_th2(val):
    global threshold2
    threshold2 = val
    process()

def on_change_acc_res(val):
    global acc_res
    acc_res = val
    process()

def on_change_min_dist(val):
    global min_dist
    min_dist = val
    process()

def on_change_min_radius(val):
    global min_radius
    min_radius = val
    process()

def on_change_max_radius(val):
    global max_radius
    max_radius = val
    process()

def on_change_seg_switch(val):
    global iris_seg_switch
    iris_seg_switch = val
    process()

def on_change_iou_min(val):
    global min_iou
    min_iou = val / 100
    process()

def on_run_starburst(val, p_center, r, tracker_name, wn = 'starburst'):
    cv2.setTrackbarPos(tracker_name, wn, 0)
    starburst(img, p_center, r)


def process():
    img_g_copy = img_grey.copy()
    no_effect = img.copy()
    cutout = no_effect.copy()
    blurred = cv2.GaussianBlur(img_g_copy, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_DEFAULT)
    blurred_2d = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    iou_scores = []
    cannied = blurred
    
    if threshold1 != 0 and threshold2 != 0:
        cannied = cv2.Canny(blurred, threshold1, threshold2)

        higher_thresh = max(threshold1, threshold2)
        rows = cannied.shape[0]

        circles = cv2.HoughCircles(
            blurred_2d, 
            cv2.HOUGH_GRADIENT, 
            acc_res, 
            rows / min_dist, 
            param1=higher_thresh, 
            param2=30, 
            minRadius=min_radius, 
            maxRadius=max_radius
        )
        cannied = cv2.merge([cannied]*3)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))

            print('-'*50)
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]

                current_score = iou(img_g_copy, filename, i)
                if current_score['score'] >= min_iou:
                    print(f'circle coor: ({current_score["x"]}, {current_score["y"]}), r: {current_score["r"]} | iou: {current_score["score"]}')

                    iou_scores.append(current_score)

                    color = (255, 0, 255) if current_score['pupil'] else (0, 255, 0)
                    # circle center
                    cv2.circle(no_effect, center, 1, (0, 100, 100), 2)
                    # circle outline
                    cv2.circle(no_effect, center, radius, color, 2)

    res = results(iou_scores)

    if iris_seg_switch == 1 and len(iou_scores) == 2 and res['precision'] == 1 and res['recall'] == 1:
        cutout = iris_segmentation(cutout, iou_scores)

    top_row = np.concatenate((no_effect, cutout), axis=1)
    bot_row = np.concatenate((blurred, cannied), axis=1)
    rows = np.concatenate((top_row, bot_row), axis=0)

    bb = write(blackboard(img_g_copy), res)

    w_blackboard = np.concatenate((bb, rows), axis=0)

    cv2.imshow(window_name, w_blackboard)

# source: https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def iou(img, img_filename, circle):
    global target_p, target_i

    if '/' in img_filename:
        img_filename = img_filename.split('/')[1]
    
    bg_result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    bg_target_i = bg_result.copy()
    bg_target_p = bg_result.copy()

    if len(target_i) == 0 or len(target_p) == 0:
        targets = pd.read_csv('data/duhovka.csv', sep=',')
        targets = targets.set_index('nazov')

        target_i = cv2.circle(bg_target_i, (targets.at[img_filename, 'dx'], targets.at[img_filename, 'dy']), targets.at[img_filename, 'dp'], (255,255,255), -1)
        target_p = cv2.circle(bg_target_p, (targets.at[img_filename, 'zx'], targets.at[img_filename, 'zy']), targets.at[img_filename, 'zp'], (255,255,255), -1)

    
    result = cv2.circle(bg_result, (circle[0], circle[1]), circle[2], (255,255,255), -1)

    # score for iris
    intersection_i = np.logical_and(target_i, result)
    union_i = np.logical_or(target_i, result)
    iou_score_i = np.sum(intersection_i) / np.sum(union_i)

    # score for pupil
    intersection_p = np.logical_and(target_p, result)
    union_p = np.logical_or(target_p, result)
    iou_score_p = np.sum(intersection_p) / np.sum(union_p)

    if iou_score_i > iou_score_p:
        return {'x': circle[0], 'y': circle[1], 'r': circle[2], 'pupil': False, 'score': iou_score_i}
    
    return {'x': circle[0], 'y': circle[1], 'r': circle[2], 'pupil': True, 'score': iou_score_p}


def blackboard(img):
    return np.zeros((img.shape[0] // 2, img.shape[1] * 2, 3), np.uint8)


def write(img, results):
    global sigma, kernel_size, threshold1, threshold2, acc_res, min_dist, min_radius, max_radius

    font = cv2.FONT_HERSHEY_SIMPLEX
    colour = (255, 255, 255)
    thickness = 1

    if img.shape[1] < 900:
        fontscale = 0.5
    elif img.shape[1] < 1200:
        fontscale = 0.75
    else: fontscale = 1

    # gaus
    y = 30
    image = cv2.putText(img, f'gaussian blur | sigma: {sigma}, kernel size: ({kernel_size},{kernel_size})', (10, int(y)), font, fontscale, colour, thickness, cv2.LINE_AA)
    
    # canny 
    y += fontscale*40
    image = cv2.putText(image, f'canny algorithm | threshold1: {threshold1}, threshold2: {threshold2}', (10, int(y)), font, fontscale, colour, thickness, cv2.LINE_AA)
    
    # hough
    y += fontscale*40
    image = cv2.putText(image, f'hough transform | acc resolution: {acc_res}, min dist. centers: {min_dist}', (10, int(y)), font, fontscale, colour, thickness, cv2.LINE_AA)
    y += fontscale*40
    image = cv2.putText(image, f'                | min radius: {min_radius}, max radius: {max_radius}', (35, int(y)), font, fontscale, colour, thickness, cv2.LINE_AA)

    if results is not None:
        y += fontscale*40
        image = cv2.putText(image, f'results | tp: {results["tp"]}, fp: {results["fp"]}, fn: {results["fn"]} | precision: {results["precision"]}, recall: {results["recall"]}', (10, int(y)), font, fontscale, colour, thickness, cv2.LINE_AA)

    return image


def results(iou_scores):
    if len(iou_scores) == 0:
        return None

    tp = 0
    fp = 0
    fn = 0
    pupil = False
    iris = False

    for i in iou_scores:
        if i['pupil']:
            if not pupil:
                tp += 1
                pupil = True
            else:
                fp += 1
        else:
            if not iris:
                tp += 1
                iris = True
            else:
                fp += 1

    if not pupil and not iris:
        fn += 2
    
    elif not pupil or not iris:
        fn += 1

    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp/(tp+fp), 'recall': tp/(tp+fn)}


def iris_segmentation(cutout, scores):
    white = (255,255,255)
    black = (0,0,0)

    iris_ind = 0
    pupil_ind = 0
    if scores[1]['pupil']:
        pupil_ind = 1
    else:
        iris_ind = 1
    
    stencil = np.zeros(cutout.shape, np.uint8)
    stencil = cv2.circle(stencil, (scores[iris_ind]['x'], scores[iris_ind]['y']), scores[iris_ind]['r'], white, -1)
    stencil = cv2.circle(stencil, (scores[pupil_ind]['x'], scores[pupil_ind]['y']), scores[pupil_ind]['r'], black, -1)

    result = cv2.bitwise_and(cutout, stencil)

    return result


def starburst(sb_orig, sb_blur, p_center):
    img = sb_blur

    length = img.shape[1]
    d = 10
    threshold = 20
    n_lines = 18
    angle_step = 360/n_lines
    angle = 0

    sobel_v = np.array([
        [-1, 0,	1],
        [-2, 0,	2],
        [-1, 0, 1]
    ])
    sobel_h = sobel_v.transpose()

    start = p_center # manually set
    prev_start = (length, length)
    print(start)
    feature_points = []
    feature_points_s1 = []
    feature_points_iter = []

    iters = 0
    while math.hypot(start[0] - prev_start[0], start[1] - prev_start[1]) > d and iters < 10:
        # stage 1
        # iterate over lines
        for i in range(n_lines):
            if i > 0:
                angle += angle_step
            
            # get last point
            end_x = int(round(start[0] + length * math.cos(angle * np.pi / 180.0)))
            end_y = int(round(start[1] + length * math.sin(angle * np.pi / 180.0)))
            end = (end_x, end_y)

            # find all points between start & end
            discrete_line = list(zip(*draw.line(*start, *end)))

            for index, point_in_line in enumerate(discrete_line):
                x = point_in_line[0]
                y = point_in_line[1]

                # check edges
                if x > img.shape[0] - 2 or x < 1:
                    break

                if y > img.shape[1] - 2 or y < 1:
                    break

                sobel_v_score = sobel_score(point_in_line, img, sobel_v)
                sobel_h_score = sobel_score(point_in_line, img, sobel_h)
                grad_intensity = math.sqrt((sobel_v_score ** 2) + (sobel_h_score ** 2))

                if index > 0:
                    grad_diff = grad_intensity - prev_grad_intensity
                    if (grad_diff > threshold):
                        ret_rays = (5*threshold) // grad_diff

                        feature_points.append(point_in_line)
                        feature_points_iter.append(point_in_line)
                        feature_points_s1.append({'point': point_in_line, 'angle': -angle, 'n_ret_rays': ret_rays if ret_rays > 5 else 5}) # + 180
                        break
                
                prev_grad_intensity = grad_intensity
         
        # stage 2
        for fp in feature_points_s1:
            start = fp['point']

            feature_points_iter.append(start)
            n_rays = fp['n_ret_rays']

            angle_step = 100 / n_rays
            angle = fp['angle'] - 50

            for i in range(n_rays):
                if i > 0:
                    angle += angle_step

                end_x = int(round(start[0] + length * math.cos(angle * np.pi / 180.0)))
                end_y = int(round(start[1] + length * math.sin(angle * np.pi / 180.0)))
                end = (end_x, end_y)
                # print((angle + (angle_step * i)) * np.pi / 180.0)
                
                # find all points between start & end
                discrete_line = list(zip(*draw.line(*start, *end)))

                for index, point_in_line in enumerate(discrete_line):
                    x = point_in_line[0]
                    y = point_in_line[1]

                    if x > img.shape[0] - 2 or x < 1:
                        break

                    if y > img.shape[1] - 2 or y < 1:
                        break

                    sobel_v_score = sobel_score(point_in_line, img, sobel_v)
                    sobel_h_score = sobel_score(point_in_line, img, sobel_h)
                    grad_intensity = math.sqrt((sobel_v_score ** 2) + (sobel_h_score ** 2))

                    if index > 0:
                        grad_diff = grad_intensity - prev_grad_intensity
                        if (grad_diff > threshold):
                            feature_points.append(point_in_line)
                            feature_points_iter.append(point_in_line)
                            break
                    
                    prev_grad_intensity = grad_intensity

        prev_start = start

        # geom. stred
        y, x = zip(*feature_points_iter)
        start = (max(x) + min(x)) // 2, (max(y) + min(y)) // 2

        feature_points_s1 = []
        feature_points_iter = []
        angle = 0
        angle_step = 360/n_lines
        iters += 1


    for i in feature_points:
        res_img = cv2.circle(sb_orig, (i[1], i[0]), 2, (0,255,0), -1)

    return res_img

    
def sobel_score(point, img, sobel):
    s_size = len(sobel)
    s_half = s_size // 2
    x = point[0]
    y = point[1]
    sobel_score = 0

    for i in range(s_size):
        for j in range(s_size):
            sobel_score += img[x - s_half + i][y - s_half + j] * sobel[i][j]
    
    return sobel_score


def process_sb(sb_blur, s, ks):
    return cv2.GaussianBlur(sb_blur, (ks, ks), s, borderType=cv2.BORDER_DEFAULT)


def starburst_app(filename):
    starburst_winname = 'starburst'
    cv2.namedWindow(starburst_winname)

    sb_orig = cv2.imread(filename)
    img_d = sb_orig
    sb_grey = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    sec_part = filename.split('/')[1]
    targets = pd.read_csv('data/duhovka.csv', sep=',')
    targets = targets.set_index('nazov')

    p_center = (targets.at[sec_part, 'zx'], targets.at[sec_part, 'zy'])
    r = targets.at[sec_part, 'zp']

    sb_grey = cv2.circle(sb_grey, p_center, r, (0,0,0), -1)

    ks = 5
    s = 20

    ks_tracker_name = 'kernel size:'
    cv2.createTrackbar(ks_tracker_name, starburst_winname, 1, 19, nothing) #lambda x: on_change_ks(x, img, sigma)
    cv2.setTrackbarMin(ks_tracker_name, starburst_winname, 1)
    cv2.setTrackbarPos(ks_tracker_name, starburst_winname, ks)

    # sigma
    sigma_tracker_name = 'sigma:'
    # cv2.createTrackbar('test', starburst_winname)
    cv2.createTrackbar(sigma_tracker_name, starburst_winname, 1, 100, nothing)
    cv2.setTrackbarMin(sigma_tracker_name, starburst_winname, 1)
    cv2.setTrackbarPos(sigma_tracker_name, starburst_winname, s)

    while True:
        sb_blur = sb_grey.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        


        s = cv2.getTrackbarPos(sigma_tracker_name, starburst_winname) / 10
        ks = cv2.getTrackbarPos(ks_tracker_name, starburst_winname)

        if ks % 2 == 0:
            ks += 1
            cv2.setTrackbarPos(ks_tracker_name, starburst_winname, ks)
        
        sb_blur = process_sb(sb_blur, s, ks)

        if key == ord('s'):
            img_d = starburst(sb_orig, sb_blur, p_center)

        stack_blur = cv2.merge([sb_blur]*3)
        res_img = np.concatenate((stack_blur, img_d), axis=1)
        cv2.imshow(starburst_winname, res_img)

    cv2.destroyAllWindows() # destroys the window showing image

    
def nothing(val):
    print(val)

def save(filename, trackbar_names):
    trackbar_values = [cv2.getTrackbarPos(i, window_name) for i in trackbar_names]
    print(trackbar_names, trackbar_values)
    df = pd.DataFrame([trackbar_values], columns=trackbar_names)

    df.to_csv(f'{filename}-best-settings.csv', index=False, mode='a')


def app(img_grey, img):
    cv2.namedWindow(window_name)
    
    imgs = np.concatenate((img, img_grey), axis=1)

    imgs_blackboard = np.concatenate((blackboard(img), imgs), axis=0)
    cv2.imshow(window_name, imgs_blackboard)

    # gaussian blur trackbars
    # kernel size
    ks_tracker_name = 'kernel size:'
    cv2.createTrackbar(ks_tracker_name, window_name, 1, 19, lambda x: on_change_ks(x, ks_tracker_name)) #lambda x: on_change_ks(x, img, sigma)
    cv2.setTrackbarMin(ks_tracker_name, window_name, 1)

    # sigma
    sigma_tracker_name = 'sigma:'
    cv2.createTrackbar(sigma_tracker_name, window_name, 1, 100, on_change_sigma)
    cv2.setTrackbarMin(sigma_tracker_name, window_name, 1)

    # canny trackbars
    # threshold 1
    th1_name = 'threshold 1:'
    cv2.createTrackbar(th1_name, window_name, 0, 255, on_change_th1) 

    # threshold 2
    th2_name = 'threshold 2:'
    cv2.createTrackbar(th2_name, window_name, 0, 255, on_change_th2) 

    # hough trackbars
    # accumulator resolution
    acc_res_name = 'accumulator resolution:'
    cv2.createTrackbar(acc_res_name, window_name, 1, 10, on_change_acc_res) 
    cv2.setTrackbarMin(acc_res_name, window_name, 1)

    # min center distance
    min_dist_name = 'centers minimum distance:'
    cv2.createTrackbar(min_dist_name, window_name, 1, 100, on_change_min_dist) 
    cv2.setTrackbarMin(min_dist_name, window_name, 1)

    # min radius
    min_radius_name = 'minimum radius:'
    cv2.createTrackbar(min_radius_name, window_name, 10, 200, on_change_min_radius) 
    cv2.setTrackbarMin(min_radius_name, window_name, 10)

    # max radius
    max_radius_name = 'maximum radius:'
    cv2.createTrackbar(max_radius_name, window_name, 10, 255, on_change_max_radius) 
    cv2.setTrackbarMin(max_radius_name, window_name, 10)

    iris_seg_switch_name = 'iris seg. switch'
    cv2.createTrackbar(iris_seg_switch_name, window_name, 0, 1, on_change_seg_switch)

    iou_min_name = 'min iou'
    cv2.createTrackbar(iou_min_name, window_name, 0, 100, on_change_iou_min)
    cv2.setTrackbarPos(iou_min_name, window_name, 75)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        elif k == ord('s'):
            save(filename, [ks_tracker_name, sigma_tracker_name, th1_name, th2_name, acc_res_name, min_dist_name, min_radius_name, max_radius_name])

    cv2.destroyAllWindows()



if __name__ == '__main__':
    filename = 'data/eye3.jpg'
    # filename = 'data/eye2.bmp'
    # filename = 'data/eye1.jpg'
    
    img_grey, img = load(filename)
    

    # app(img_grey, img)

    starburst_app(filename)

    
