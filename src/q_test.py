import numpy as np
import chainer
from chainer import cuda, Variable, serializers, functions as F
from net import ImageCaption
import argparse
import pickle
import json
from tqdm import tqdm


def create_data(all_data, all_features, split_type):
    features = []
    qa_ids = []
    target_vecs = []

    for each_question in all_data:
        if each_question['split'] == split_type:
            features.append(all_features[each_question['qa_id']])
            qa_ids.append(each_question['qa_id'])
            target_vecs.append(each_question['target_vec'])
    return np.array(features, dtype=np.float32), qa_ids, np.array(target_vecs, dtype=np.float32)


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


def main(args):
    model_path = args.MODEL_PATH
    all_data_path = args.ALL_DATA_PATH
    all_feature_path = args.ALL_FEATURE_PATH
    save_path = args.SAVE_PATH

    with open(args.WORD2ID_PATH, 'rb') as f:
        word_id_dic = pickle.load(f)

    bos = word_id_dic['<s>']
    eos = word_id_dic['</s>']
    unk = word_id_dic['<unk>']


    id2word = {word_id_dic[x]: x for x in word_id_dic.keys()}

    print('data loading...')
    with open(all_data_path, 'rb') as f:
        all_data = pickle.load(f)
    with open(all_feature_path, 'rb') as f:
        all_features = pickle.load(f)

    print('data loaded!')

    test_features, test_qa_ids, test_target_vecs = create_data(all_data, all_features, 'test')

    feature_num = 2005
    hidden_num = 1024
    vocab_num = len(word_id_dic)
    attr_num = 5

    CaptionNet = ImageCaption(vocab_num, attr_num, feature_num, hidden_num)
    serializers.load_hdf5(model_path, CaptionNet)
    CaptionNet.to_gpu(gpu_device)

    beam_width = 3
    max_length = 100

    question_list = []

    for i in tqdm(range(len(test_qa_ids))):
        qa_id = test_qa_ids[i]
        target_vec = test_target_vecs[i]

        target_var = Variable(xp.array([target_vec], dtype=xp.float32))
        concat_feature = test_features[i]
        feature_var = Variable(xp.array([concat_feature], dtype=xp.float32))

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            CaptionNet.image_init(feature_var)

        candidates = [(CaptionNet, [bos], 0)]
        next_candidates = beam_search(candidates, target_var, norm=True)
        for j in range(max_length):
            next_candidates = beam_search(next_candidates, target_var, norm=True)
            if all([x[1][-1] == eos for x in next_candidates]):
                break
        result = [k[1] for k in next_candidates]
        tokens = [id2word[token_id] for token_id in result[0][1:-1]]
        question_list.append([qa_id, tokens])

    all_list = []
    for each_question in question_list:
        each_dic = {}
        qa_id = each_question[0]
        question = each_question[1]
        join_question = (' '.join([word + '' for word in question]) + '?').capitalize()
        each_dic['id'] = qa_id
        each_dic['image_id'] = qa_id
        each_dic['caption'] = join_question
        all_list.append(each_dic)

    with open(save_path, 'w') as f:
        json.dump(all_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--GPU', type=int, default=0)
    parser.add_argument('-model', '--MODEL_PATH', required=True)
    parser.add_argument('-word2id', '--WORD2ID_PATH', default='../data/word2id.pickle')
    parser.add_argument('-all_data', '--ALL_DATA_PATH', default='../data/processed_data.pickle')
    parser.add_argument('-all_feature', '--ALL_FEATURE_PATH', default='../data/all_feature.pickle')
    parser.add_argument('-save', '--SAVE_PATH', default='../data/q_result.json')
    args = parser.parse_args()

    gpu_device = args.GPU

    cuda.get_device_from_id(gpu_device).use()
    xp = cuda.cupy

    main(args)
