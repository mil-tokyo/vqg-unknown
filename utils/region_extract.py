import itertools
import numpy as np
from skimage import io
import cv2
import chainer
from chainer import cuda, Variable
import chainer.functions as F
from chainer.links.model.vision import resnet
import selectivesearch
import pyimgsaliency as psal

xp = cuda.cupy


def selective_regions(image, saliency_image):
    # perform selective search and return regions which area is less 90% of image area
    otsu, thres = cv2.threshold(saliency_image, 0, 255, cv2.THRESH_OTSU)

    saliency_image[saliency_image < otsu] = 0
    saliency_image[saliency_image >= otsu] = 1

    combine_image = image.copy()
    combine_image[:, :, 0] = image[:, :, 0] * saliency_image
    combine_image[:, :, 1] = image[:, :, 1] * saliency_image
    combine_image[:, :, 2] = image[:, :, 2] * saliency_image

    image = combine_image

    img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=10)
    image_area = image.shape[0] * image.shape[1]
    regions = [x['rect'] for x in regions if
               image_area * 0.01 < (x['rect'][2] * x['rect'][3]) < image_area * 0.9]

    return regions


def crop_image(image, regions):
    cropped_image = []
    for each_region in regions:
        x, y, w, h = each_region
        cropped_image.append(image[y: y+h, x: x+w, :])
    return cropped_image


def pred(image, model):
    image = resnet.prepare(image, size=None)
    x_data = Variable(xp.array([image]))
    with chainer.using_config('train', False):
        prob = cuda.to_cpu(F.softmax(model(x_data)).data)
    return prob


def classify(regions, probs, thres):
    region_list = []
    for i, each_prob in enumerate(probs):
        entropy = calc_entropy(each_prob)
        if entropy > thres:
            label = 1000
            region_list.append([regions[i], label, entropy, 0])
        else:
            label = np.argmax(each_prob)
            region_list.append([regions[i], label, entropy, 0])
    return region_list


def calc_entropy(prob):
    entropy = -np.sum(prob * np.log2(prob + np.array(1e-20)))
    return entropy


def IoU(label_combi):
    x0, y0, w0, h0 = label_combi[0][0]
    x1, y1, w1, h1 = label_combi[1][0]
    intersect_y = np.intersect1d(np.arange(y0, y0 + h0), np.arange(y1, y1 + h1)).shape[0]
    intersect_x = np.intersect1d(np.arange(x0, x0 + w0), np.arange(x1, x1 + w1)).shape[0]
    area_overlap = intersect_x * intersect_y
    area_union = w0 * h0 + w1 * h1 - area_overlap
    return area_overlap / area_union


def nms(label_regions, iou_thres):
    label_dic = {}
    for each_region in label_regions:
        if each_region[1] in label_dic:
            label_dic[each_region[1]].append(each_region)
        else:
            label_dic[each_region[1]] = []
    for each_label in label_dic:
        label_combi = list(itertools.combinations(label_dic[each_label], 2))
        for each_combi in label_combi:
            iou = IoU(each_combi)
            if iou >= iou_thres:
                if each_combi[0][2] >= each_combi[1][2]:
                    each_combi[1][3] = 1
                else:
                    each_combi[0][3] = 1
    result_list = []
    for each_label in label_dic:
        for each_data in label_dic[each_label]:
            if each_data[3] == 0:
                result_list.append(each_data)
    return result_list


def calc_region_saliency(saliency_image, region, thres):
    x, y, w, h = region
    saliency_region = saliency_image[y: y+h, x: x+w]
    sum_saliency = np.sum(saliency_region[saliency_region > thres])
    area = saliency_region.shape[0] * saliency_region.shape[1]
    area_saliency = saliency_region[saliency_region > thres].shape[0]
    return sum_saliency * (area_saliency / (area ** 1.5))


def region_extract(image_path, model, ent_thres=1.75):
    image = io.imread(image_path)

    salient_image = psal.get_saliency_rbd(image_path).astype('uint8')
    thres, thresed_image = cv2.threshold(salient_image, 0, 255, cv2.THRESH_OTSU)

    regions = selective_regions(image, salient_image)
    crop_images = crop_image(image, regions)

    probs = []
    for each_region in crop_images:
        each_prob = pred(each_region, model)
        probs.append(each_prob[0])
    probs = np.asarray(probs)

    entropy_list = classify(regions, probs, ent_thres)
    regions = nms(entropy_list, 0.5)

    region_saliency_list = []
    for each_region in regions:
        if each_region[1] == 1000:
            region_saliency = calc_region_saliency(salient_image, each_region[0], thres)
            region_saliency_list.append(region_saliency)
    target_region = regions[np.argsort(np.asarray(region_saliency_list))[::-1][0]][0]

    return target_region
