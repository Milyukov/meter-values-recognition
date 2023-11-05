import cv2
import numpy as np

from shapely.geometry import Polygon

def resize_image_keypoints(im, keypoints, width, height):
    h, w, c = im.shape
    delta_top = 0
    delta_bottom = 0
    delta_left = 0
    delta_right = 0
    if h > w:
        delta = h - w
        delta_left = delta // 2
        delta_right = delta // 2
        if delta % 2 != 0:
            delta_right += 1
    else:
        delta = w - h
        delta_top = delta // 2
        delta_bottom = delta // 2
        if delta % 2 != 0:
            delta_top += 1
    keypoints_resized = []
    for point in keypoints:
        keypoints_resized.append([point[0], point[1]])
    for point in keypoints_resized:
        point[0] += delta_left
        point[1] += delta_top
    im_resized = cv2.copyMakeBorder(
        im, delta_top, delta_bottom, delta_left, delta_right, cv2.BORDER_CONSTANT, value=0)
    side = im_resized.shape[0]
    scale = 1024.0 / side
    for point_idx, point in enumerate(keypoints_resized):
        keypoints_resized[point_idx] = (point[0] * scale, point[1] * scale)
    im_resized = cv2.resize(im_resized, (width, height))
    keypoints_resized = np.array(keypoints_resized)
    return im_resized, keypoints_resized

def resize_image_bboxes(im, bboxes, width, height):
    h, w, c = im.shape
    delta_top = 0
    delta_bottom = 0
    delta_left = 0
    delta_right = 0
    # make image rectangular
    if h > w:
        delta = h - w
        delta_left = delta // 2
        delta_right = delta // 2
        if delta % 2 != 0:
            delta_right += 1
    else:
        delta = w - h
        delta_top = delta // 2
        delta_bottom = delta // 2
        if delta % 2 != 0:
            delta_top += 1
    bboxes_resized = []
    for box in bboxes:
        bboxes_resized.append(box[:])
    for box in bboxes_resized:
        box[0] += delta_left
        box[1] += delta_top
    im_resized = cv2.copyMakeBorder(
        im, delta_top, delta_bottom, delta_left, delta_right, cv2.BORDER_CONSTANT, value=0)
    side = im_resized.shape[0]
    scale = height / side
    for box_idx, box in enumerate(bboxes_resized):
        bboxes_resized[box_idx] = [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]
    im_resized = cv2.resize(im_resized, (width, height))
    bboxes_resized = np.array(bboxes_resized)
    return im_resized, bboxes_resized

def sort_keypoints(keypoints):
    # assumption: left points are on the left
    keypoints_sorted = sorted(keypoints, key=lambda x: x[0])
    if keypoints_sorted[0][1] < keypoints_sorted[1][1]:
        top_left = keypoints_sorted[0]
        bottom_left = keypoints_sorted[1]
    else:
        top_left = keypoints_sorted[1]
        bottom_left = keypoints_sorted[0]
    if keypoints_sorted[2][1] < keypoints_sorted[3][1]:
        top_right = keypoints_sorted[2]
        bottom_right = keypoints_sorted[3]
    else:
        top_right = keypoints_sorted[3]
        bottom_right = keypoints_sorted[2]
    return np.array([top_left, top_right, bottom_right, bottom_left])

def process_keypoints(im, keypoints, width, height):
    # sort keypoints: top-left, top-right, bottom-rightm bottom-left
    #keypoints = sort_keypoints(keypoints) # they are sorted by design
    # resize image and keypoints
    im_resized, keypoints_resized = resize_image_keypoints(im, keypoints, width, height)
    # create bounding box
    bbox = cv2.boundingRect(keypoints_resized.astype(np.int32))
    bbox = np.array(bbox)
    return im_resized, bbox, keypoints_resized 

def process_bboxes(im, bboxes, width, height):
    # resize image and bboxes
    im_resized, bboxes_resized = resize_image_bboxes(im, bboxes, width, height)
    return im_resized, bboxes_resized 
    

def extract_rectangle_area(im_resized, bbox, keypoints):
    # evaluate homography transform to warp and crop image inside keypoints
    # crop keypoints coordinates
    x_min = np.min(keypoints[::2])
    y_min = np.min(keypoints[1::2])
    keypoints = np.array([
        [keypoints[0], keypoints[1]], 
        [keypoints[2], keypoints[3]], 
        [keypoints[4], keypoints[5]], 
        [keypoints[6], keypoints[7]]])
    keypoints[:, 0] -= x_min
    keypoints[:, 1] -= y_min
    width = int(np.sqrt((keypoints[0][0] - keypoints[1][0]) ** 2 + (keypoints[0][1] - keypoints[1][1]) ** 2))
    height = int(np.sqrt((keypoints[0][0] - keypoints[3][0]) ** 2 + (keypoints[0][1] - keypoints[3][1]) ** 2))
    keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)

    h, status = cv2.findHomography(keypoints.astype(np.int32), keypoints_planar)
    # for extended area
    h_inv, status = cv2.findHomography(keypoints_planar, keypoints.astype(np.int32))

    # extend area
    keypoints_planar_extended = keypoints_planar + np.array(
        [[-height//2, -2 * height//3], 
        [height//2, -2 * height//3], 
        [height//2, 2 * height//3], 
        [-height//2, 2 * height//3]])
    
    # generate ROI inside cropped region
    roi = np.array([
        [height//2, height//2],
        [width + height//2, height//2],
        [width + height//2, height + height//2],
        [height//2, height + height//2],
    ])

    # use inverse homography to choose new points on the original image
    keypoints_planar_extended = keypoints_planar_extended.reshape(-1,1,2).astype(np.float32)
    keypoints_extended = cv2.perspectiveTransform(keypoints_planar_extended, h_inv)
    keypoints_extended = keypoints_extended.reshape(-1, 2)
    keypoints_extended[:, 0] += x_min
    keypoints_extended[:, 1] += y_min

    # get bbox
    bbox = cv2.boundingRect(keypoints_extended.astype(np.int32))
    width = int(np.sqrt((keypoints_extended[0][0] - keypoints_extended[1][0]) ** 2 + (keypoints_extended[0][1] - keypoints_extended[1][1]) ** 2))
    height = int(np.sqrt((keypoints_extended[1][0] - keypoints_extended[2][0]) ** 2 + (keypoints_extended[1][1] - keypoints_extended[2][1]) ** 2))

    # compute new homography matrix for extended area
    x_min = np.min(keypoints_extended[:, 0])
    y_min = np.min(keypoints_extended[:, 1])
    keypoints_extended[:, 0] -= x_min
    keypoints_extended[:, 1] -= y_min
    keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
    h, status = cv2.findHomography(keypoints_extended.astype(np.int32), keypoints_planar)
    keypoints_extended[:, 0] += x_min
    keypoints_extended[:, 1] += y_min

    # warp image area
    # extend image if bbox is out of image's plane
    top = 0 if bbox[1] > 0 else -bbox[1]
    bottom = 0 if bbox[3] < im_resized.shape[0] else bbox[3] - im_resized.shape[0]
    left = 0 if bbox[0] > 0 else -bbox[0]
    right = 0 if bbox[2] < im_resized.shape[1] else bbox[2] - im_resized.shape[1]
    im_resized = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    im_dst = cv2.warpPerspective(im_resized[bbox[1] + top:bbox[1] + top + bbox[3], 
                                            bbox[0] + left:bbox[0] + left + bbox[2]], h, (width, height))
    return im_dst, roi

def parse_analog_detection(boxes, scores, class_names, roi=None):
    kept_indices = []
    kept_boxes = []
    kept_scores = []
    kept_class_names = []
    digits_after_fpoint = []
    # filter out areas after point except the largest one
    area = 0
    largest_index = -1
    indecies_to_remove = []
    for box_index, (box, _cls, score) in enumerate(zip(boxes, class_names, scores)):
        if _cls == '14':
            indecies_to_remove.append(box_index)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if area < abs(w * h):
                area = w * h
                largest_index = box_index
    if len(indecies_to_remove) > 0:
        indecies_to_remove.remove(largest_index)
    for box_index1, (box1, _cls1, score1) in enumerate(zip(boxes, class_names, scores)):
        if _cls1 == '14':
            if box_index1 in indecies_to_remove:
                continue
        duplicated = False
        x11, y11, x12, y12 = box1
        w1, h1 = x12 - x11, y12 - y11
        polygon1 = Polygon([[x11, y11], 
                            [x12, y11], 
                            [x12, y12],
                            [x11, y12]])
        
        if roi is not None:
            ul, ur, br, bl = roi
            xr1, yr1 = ul
            xr2, yr2 = br
            polygon_roi = Polygon([[xr1, yr1], 
                            [xr2, yr1], 
                            [xr2, yr2],
                            [xr1, yr2]])
            intersection = polygon1.intersection(polygon_roi).area
            if intersection == 0:
                continue

        for box_index2, (box2, _cls2, score2) in enumerate(zip(boxes, class_names, scores)):
            if box_index1 == box_index2:
                continue
            x21, y21, x22, y22 = box2
            w2, h2 = x22 - x21, y22 - y21
            polygon2 = Polygon([[x21, y21], 
                                [x22, y21], 
                                [x22, y22],
                                [x21, y22]])
            intersection = polygon1.intersection(polygon2).area
            union = polygon1.union(polygon2).area
            iou = intersection / union
            if _cls2 == '14':
                if box_index2 in indecies_to_remove:
                    continue
                if intersection > 0.2 * polygon1.area:
                    digits_after_fpoint.append(box_index1)
            elif iou > 0.9:
                if score1 > score2:
                    if not box_index1 in kept_indices:
                        kept_indices.append(box_index1)
                    duplicated = True
                else:
                    if not box_index2 in kept_indices:
                        kept_indices.append(box_index2)
                    duplicated = True
                break
            else:
                x_intersection = min(x12, x22) - max(x11, x21)
                # case when two digits - one above - another are detected
                # choose the one with the largest area
                if x_intersection > 0.2 * abs(w1):
                    if polygon1.area > polygon2.area:
                        if not box_index1 in kept_indices:
                            kept_indices.append(box_index1)
                        duplicated = True
                    else:
                        if not box_index2 in kept_indices:
                            kept_indices.append(box_index2)
                        duplicated = True
        if not duplicated:
            kept_indices.append(box_index1)
    kept_boxes = boxes[kept_indices]
    kept_scores = scores[kept_indices]
    kept_class_names = np.array(class_names)[kept_indices]
    one_hot_digits_after_fpoint = np.array([1 if i in digits_after_fpoint else 0 for i in range(len(boxes))])
    one_hot_digits_after_fpoint = one_hot_digits_after_fpoint[kept_indices]
    # sort by x-coordinate
    indices = np.argsort(kept_boxes[:, 0])
    xs = kept_boxes[indices][:, 0]
    dist_between_digits = np.median(np.diff(xs))
    one_hot_digits_after_fpoint = one_hot_digits_after_fpoint[indices]
    fpoint_pos = -1
    if np.any(one_hot_digits_after_fpoint > 0):
        fpoint_pos = (one_hot_digits_after_fpoint > 0).argmax()
    text = ''
    for index in range(len(kept_class_names)):
        if kept_class_names[indices][index] == '14':
            continue
        if index == fpoint_pos:
            text += '.'
        if index == 0:
            text += kept_class_names[indices][index]
        else:
            num_of_unknown = (xs[index] - xs[index - 1]) / dist_between_digits
            num_of_unknown = int(num_of_unknown + 0.5) - 1
            text += 'x' * num_of_unknown + kept_class_names[indices][index]
    return text, kept_boxes, kept_scores, kept_class_names
            
