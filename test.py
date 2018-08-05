from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.region_extract import *
from utils.utils import *
from src.net import ResNet
from src.net import ImageCaption

import chainer
from chainer import Variable, serializers, cuda
import chainer.functions as F
from nltk.corpus import wordnet as wn
import argparse
import pickle
from skimage import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-label', '--LABEL_PATH', default='./data/synset_words_caffe_ILSVRC12.txt')
    parser.add_argument('-emb', '--WORD_EMB', default='./data/word_embeddings.pickle')
    parser.add_argument('-id', '--WORD_ID', default='./data/word2id.pickle')
    parser.add_argument('-g', '--GPU_ID', type=int, default=0)
    parser.add_argument('-cnn', '--CNN_PATH', default='./data/resnet_152.caffemodel')
    parser.add_argument('-model', '--MODEL_PATH', default='./data/pretrained_model.h5')
    parser.add_argument('-image', '--IMAGE_PATH', default='./data/test.jpg')
    parser.add_argument('-save', '--SAVE_FIG', default='False')
    parser.add_argument('-hyper', '--HYPER_NUM', type=int, default=2)
    args = parser.parse_args()

    if args.SAVE_FIG == 'False':
        save_flag = False
    else:
        save_flag = True

    label_synset_list = open(args.LABEL_PATH).read().split('\n')
    del label_synset_list[-1]
    label_list = [x.split()[0] for x in label_synset_list]

    with open(args.WORD_EMB, 'rb') as f:
        target2vec = pickle.load(f)
    with open(args.WORD_ID, 'rb') as f:
        word2id = pickle.load(f)

    id2word = {word2id[x]: x for x in word2id.keys()}

    bos = word2id['<s>']
    eos = word2id['</s>']

    feature_num = 2005
    hidden_num = 1024
    vocab_num = len(word2id)
    attr_num = 5
    max_length = 100

    gpu_device = args.GPU_ID
    cuda.get_device_from_id(gpu_device).use()
    cnn_model = ResNet(args.CNN_PATH, 152)
    cnn_model.to_gpu(gpu_device)

    CaptionNet = ImageCaption(vocab_num, attr_num, feature_num, hidden_num)
    serializers.load_hdf5(args.MODEL_PATH, CaptionNet)
    CaptionNet.to_gpu(gpu_device)

    image = io.imread(args.IMAGE_PATH)
    target_region = region_extract(args.IMAGE_PATH, cnn_model)
    x, y, w, h = target_region
    region_image = image[y: y + h, x: x + w, :]

    entire_feature = cuda.to_cpu(extract_feature(image, cnn_model).data)
    region_feature = cuda.to_cpu(extract_feature(region_image, cnn_model).data)

    location_vector = locate_feature(image, target_region)
    concat_feature = np.concatenate([region_feature[0], entire_feature[0], location_vector], axis=0)

    prob = F.softmax(region_feature)
    label_index = np.argsort(cuda.to_cpu(prob.data)[0])[::-1]
    pred_synset_list = [wn._synset_from_pos_and_offset('n', label_convert(label_list[x])) for x in label_index[:4]]
    synset_list = pred_synset_list[:args.HYPER_NUM + 1]
    common_synset = common_hypernyms(synset_list)
    target_synset = common_synset.name()

    try:
        target_vec = target2vec[target_synset]
    except:
        target_vec = np.mean(np.asarray(list(target2vec.values())), axis=0)

    target_var = Variable(xp.array([target_vec], dtype=xp.float32))
    feature_var = Variable(xp.array([concat_feature], dtype=xp.float32))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        CaptionNet.image_init(feature_var)
    candidates = [(CaptionNet, [bos], 0)]
    next_candidates = beam_search(candidates, target_var)
    for i in range(max_length):
        next_candidates = beam_search(next_candidates, target_var)
        if all([x[1][-1] == eos for x in next_candidates]):
            break
    result = [k[1] for k in next_candidates]

    generated_question = []
    for token_ids in result:
        tokens = [id2word[token_id] for token_id in token_ids[1:-1]]
        generated_question.append(' '.join(tokens))

    if save_flag:
        fig = plt.figure()
        fig = fig.add_subplot(111)
        rect = plt.Rectangle((x, y), w, h, edgecolor='red', fill=False, linewidth=3)
        fig.add_patch(rect)
        fig.imshow(image)
        plt.show()
        plt.savefig('result.png')

    return generated_question[0]


if __name__ == '__main__':
    question = main()
    print(question)
