import cv2
import numpy as np

def resize_image(im, keypoints, width, height):
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
    keypoints = sort_keypoints(keypoints)
    # resize image and keypoints
    im_resized, keypoints = resize_image(im, keypoints, width, height)
    # create bounding box
    bbox = cv2.boundingRect(keypoints.astype(np.int32))
    # evaluate homography transform to warp and crop image inside keypoints
    # crop keypoints coordinates
    x_min = np.min(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    keypoints[:, 0] -= x_min
    keypoints[:, 1] -= y_min
    width = bbox[2]
    height = bbox[3]
    keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
    h, status = cv2.findHomography(keypoints.astype(np.int32), keypoints_planar)
    im_dst = cv2.warpPerspective(im_resized[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], h, (width, height))
    im_dst = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)
    im_dst_eq = cv2.equalizeHist(im_dst)
    return bbox, im_resized, im_dst, im_dst_eq
