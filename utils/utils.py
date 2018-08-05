import chainer
from chainer import cuda
import chainer.functions as F
from chainer.links.model.vision import resnet
import numpy as np

xp = cuda.cupy


def extract_feature(image, model):
    x = resnet.prepare(image, size=None)
    x = chainer.Variable(xp.asarray([x], dtype=xp.float32))
    with chainer.using_config('train', False):
        feature = model(x)
    return feature


def locate_feature(image, region):
    x_tl, y_tl, w_r, h_r = region
    x_br = x_tl + w_r
    y_br = y_tl + h_r
    W = image.shape[0]
    H = image.shape[0]
    S_b = w_r * h_r
    S_i = W * H

    location_vector = np.array([float(x_tl) / W, float(y_tl) / H, float(x_br) / W, float(y_br) / H, float(S_b) / S_i])
    return location_vector


def label_convert(label):
    if label[0] == "0":
        return int(label[2:])
    else:
        return int(label[1:])


def common_hypernyms(synset_list):
    if len(synset_list) == 2:
        return synset_list[0]
    else:
        common = synset_list[0].lowest_common_hypernyms(synset_list[1])
        common_list = synset_list[2:]
        return_list = common + common_list
        return common_hypernyms(return_list)


def beam_search(candidates, target_vec, norm=True, eos=1, beam_width=3):
    next_candidate = []
    for each_candidate in candidates:
        prev_net, words, prob = each_candidate
        if words[-1] == eos:
            next_candidate.extend([each_candidate])
            continue
        net = prev_net.copy()
        x = xp.asarray([words[-1]]).astype(np.int32)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            y = F.softmax(net(x, target_vec))
        ndarr_y = np.log(cuda.to_cpu(y.data[0]) + 1e-15)
        order = np.argsort(ndarr_y)[::-1][:beam_width]
        if norm:
            next_each_candidate = [(net, words + [j], ((prob * (len(words) - 1)) + ndarr_y[j]) / len(words)) for j in order]
        else:
            next_each_candidate = [(net, words + [j], (prob + ndarr_y[j])) for j in order]
        next_candidate.extend(next_each_candidate)
    sorted_candidate = sorted(next_candidate, key=lambda b: b[2], reverse=True)[:beam_width]
    return sorted_candidate
