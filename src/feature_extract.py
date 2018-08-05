import pickle
from skimage import io
from scipy.misc import imresize
import chainer
from chainer import Variable, cuda
from chainer.links.model.vision import resnet
import numpy as np
import argparse
from net import ResNet

xp = cuda.cupy


def region_resize(image, region):
    x, y, w, h = region
    aspect_list = np.array([36/1, 18/2, 12/3, 9/4, 6/6])
    size_list = np.array([[36, 1], [18, 2], [12, 3], [9, 4], [6, 6]]) * 32
    if w > h:
        region_aspect = w / h
        new_w, new_h = size_list[np.argmin(np.absolute(aspect_list - region_aspect))]
    else:
        region_aspect = h / w
        new_h, new_w = size_list[np.argmin(np.absolute(aspect_list - region_aspect))]

    new_image = imresize(image[y: y + h, x: x + w, :], (int(new_h), int(new_w)), interp='bicubic', mode='RGB')

    return new_image


def image_resize(image):
    h, w, _ = image.shape
    aspect_list = np.array([144/1, 72/2, 48/3, 36/4, 24/6, 18/8, 16/9, 12/12])
    size_list = np.array([[144, 1], [72, 2], [48, 3], [36, 4], [24, 6], [18, 8], [16, 9], [12, 12]]) * 32
    if w > h:
        region_aspect = w / h
        new_w, new_h = size_list[np.argmin(np.absolute(aspect_list - region_aspect))]
    else:
        region_aspect = h / w
        new_h, new_w = size_list[np.argmin(np.absolute(aspect_list - region_aspect))]

    new_image = imresize(image, (int(new_h), int(new_w)), interp='bicubic', mode='RGB')

    return new_image


def feature_extract(img, model):
    img = resnet.prepare(img, size=None)
    img = np.asarray([img], dtype=np.float32)
    img = Variable(cuda.to_gpu(img))
    with chainer.using_config("train", False):
        feature = cuda.to_cpu(model(img).data)
    return feature


def main():
    parser = argparse.ArgumentParser(description='extract features')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID')
    parser.add_argument('--ALL_DATA_PATH', '-all', default='../data/processed_data.pickle')
    parser.add_argument('--MODEL_PATH', '-model', default='../data/resnet_152.caffemodel')
    parser.add_argument('--SAVE_PATH', '-save', default='../data/all_feature.pickle')
    args = parser.parse_args()

    gpu_device = args.gpu
    all_data_path = args.ALL_DATA_PATH
    model_path = args.MODEL_PATH
    save_path = args.SAVE_PATH

    with open(all_data_path, 'rb') as f:
        data = pickle.load(f)

    cuda.get_device_from_id(gpu_device).use()
    model = ResNet(model_path, 152)
    model.to_gpu(gpu_device)

    all_dic = {}

    for each_question in data:
        try:
            image_path = each_question['image_path']
            image = io.imread(image_path)
            if image.ndim == 2:
                image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

            qa_id = each_question['qa_id']
            crop_region = [each_question['x'], each_question['y'], each_question['w'], each_question['h']]
            h, w, _ = image.shape
            x_region, y_region, w_region, h_region = crop_region

            resize_entire = image_resize(image)
            resize_region = region_resize(image, crop_region)

            entire_feature = feature_extract(resize_entire, model)
            region_feature = feature_extract(resize_region, model)
            concat_feature = np.concatenate((region_feature, entire_feature), axis=1)[0]

            x_tl, y_tl, x_br, y_br = x_region, y_region, x_region + w_region, y_region + h_region
            region_inf = np.asarray([x_tl/w, y_tl/h, x_br/w, y_br/h, (w_region * h_region)/(w * h)])
            concat_all = np.concatenate((concat_feature, region_inf), axis=0)

            all_dic[qa_id] = concat_all
        except:
            continue

    with open(save_path, 'wb') as f:
        pickle.dump(all_dic, f)


if __name__ == '__main__':
    main()
