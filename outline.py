import cv2
from hed import hed
import os
import PIL.Image
import PIL.ImageEnhance
import torch
import numpy as np

def get_connected_components(image, connected_threshold=25):

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, None, None, None, 8, cv2.CV_32S)

    areas = stats[1:, cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] > connected_threshold:  # keep
            result[labels == i + 1] = 255

    return result


def skeletonize(image):
    img = image.copy()
    skel = np.zeros(img.shape, np.uint8)
    size = np.size(img)
    print (size)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def increase_red(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    enhancer = PIL.ImageEnhance.Contrast(img)
    img2 = enhancer.enhance(1)
    converter = PIL.ImageEnhance.Color(img2)
    img2 = converter.enhance(2.0)

    img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
    return img2


if __name__ == '__main__':

    input_image = "brain.png"

    img = PIL.Image.open(input_image)
    img = np.array(img)[:,:,:3]

    color_image = cv2.imread(input_image)
    
    # blank_image = np.zeros(np.array(img).shape, np.uint8)

    tenInput = torch.FloatTensor(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    tenOutput = hed.estimate(tenInput)

    img = (tenOutput.clip(0.0, 1.0).numpy().transpose(
        1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)

    cv2.imshow('out', img)

    img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]

    final = cv2.bitwise_and(color_image, color_image, mask=img)

    final = increase_red(final)

    

    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.dilate(img, kernel,iterations = 1)
    # img = cv2.erode(img, kernel,iterations = 2)
    # img = get_connected_components(img)


    # contours, heirarchy = cv2.findContours(
    #     img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(blank_image,contours, -1,(0,255,0),1)
    # for contour in contours:
    #     blank_image = cv2.polylines(blank_image, [contour], -1, (0, 255, 0), 1)

    # cv2.imshow('frame', blank_image)
    # cv2.waitKey(3000)
    # blank_img = cv2.dilate(blank_image,kernel,iterations =5)
    # cv2.imshow('frame', blank_image)
    # skel = skeletonize(img)
    # skel = cv2.threshold(skel, 127, 255, cv2.THRESH_BINARY)[1]
    # skel = cv2.dilate(skel,kernel,iterations = 1)
    # skel = get_connected_components(skel, 5)


    # final = final + 20

    # cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('Thresh', img)
    cv2.imshow('Contours', final)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
