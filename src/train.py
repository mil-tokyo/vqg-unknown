import numpy as np
import chainer
from chainer import cuda, Variable, optimizers, serializers, functions as F
import pickle
import argparse
import os
from net import ImageCaption

from tqdm import tqdm


def forward(target_vec, image_features, questions, model, eos):
    sentence_length = questions.shape[-1]
    model.image_init(image_features)

    loss = 0
    acc = 0
    size = 0

    for i in range(sentence_length - 1):
        target = xp.where(xp.asarray(questions[:, i].data) != eos, 1, 0).astype(np.float32)
        if(target == 0).all():
            break
        X = Variable(xp.array(questions[:, i].data, dtype=xp.int32))
        t = Variable(xp.array(questions[:, i+1].data, dtype=xp.int32))
        y = model(X, target_vec)

        y_max_index = xp.argmax(y.data, axis=1)
        mask = target.reshape((len(target), 1)).repeat(y.data.shape[1], axis=1)
        y = y * Variable(mask)
        loss += F.softmax_cross_entropy(y, t)
        acc += xp.sum((y_max_index == t.data) * target)
        size += xp.sum(target)

    return loss, acc, size


def load_batch(batch, all_features):
    features_list = []
    questions_list = []
    targets_list = []

    for each_question in batch:
        qa_id = each_question[0]
        question = each_question[1]
        target_id = each_question[2]
        feature = all_features[qa_id]

        features_list.append(feature)
        questions_list.append(question)
        targets_list.append(target_id)
    return features_list, questions_list, targets_list


def create_data(all_data, all_features, split_type):
    features = []
    questions = []
    target_vecs = []

    for each_question in all_data:
        if each_question['split'] == split_type:
            features.append(all_features[each_question['qa_id']])
            questions.append(each_question['question_word_id'])
            target_vecs.append(each_question['target_vec'])
    return np.array(features, dtype=np.float32), np.array(questions, dtype=np.float32), np.array(target_vecs, dtype=np.float32)



def main(args):
    model_save_path = args.MODEL_PATH
    all_data_path = args.ALL_DATA_PATH
    all_feature_path = args.ALL_FEATURE_PATH

    if not os.path.exists(model_save_path):
        print('make model save directory')
        os.mkdir(model_save_path)

    with open(args.WORD2ID_PATH, 'rb') as f:
        word_id_dic = pickle.load(f)

    eos = word_id_dic['</s>']

    print('all data loading...')
    with open(all_data_path, 'rb') as f:
        all_data = pickle.load(f)
    print('all data loaded!')
    print('all feature loading...')
    with open(all_feature_path, 'rb') as f:
        all_features = pickle.load(f)
    print('all feature loaded!')

    train_features, train_questions, train_target_vecs = create_data(all_data, all_features, 'train')
    valid_features, valid_questions, valid_target_vecs = create_data(all_data, all_features, 'valid')

    feature_num = 2005
    hidden_num = 1024
    vocab_num = len(word_id_dic)
    attr_num = 5

    epoch_num = 100
    batch_size = 100

    CaptionNet = ImageCaption(vocab_num, attr_num, feature_num, hidden_num)
    CaptionNet.to_gpu(gpu_device)

    optimizer = optimizers.Adam(alpha=4.0e-4)

    optimizer.setup(CaptionNet)

    N_train = len(train_target_vecs)
    N_valid = len(valid_target_vecs)

    for epoch in tqdm(range(epoch_num)):
        train_perm = np.random.permutation(N_train)
        sum_loss = 0
        sum_acc = 0
        sum_size = 0

        for index in range(0, N_train, batch_size):
            CaptionNet.zerograds()
            index_array = train_perm[index: index+batch_size]
            batch_features = xp.array(train_features[index_array])
            batch_questions = xp.array(train_questions[index_array])
            batch_targets = xp.array(train_target_vecs[index_array])

            feature_var = Variable(batch_features)
            question_var = Variable(batch_questions)
            target_var = Variable(batch_targets)

            loss, acc, size = forward(target_var, feature_var, question_var, CaptionNet, eos)

            loss.backward()
            optimizer.update()
            sum_loss += loss.data.tolist()
            sum_acc += acc.tolist()
            sum_size += size.tolist()

        print('loss:', sum_loss / sum_size, 'acc:', sum_acc / sum_size)

        valid_sum_loss = 0
        valid_sum_acc = 0
        valid_sum_size = 0

        for valid_index in range(0, N_valid, batch_size):
            valid_batch_features = valid_features[valid_index: valid_index+batch_size]
            valid_batch_questions = valid_questions[valid_index: valid_index+batch_size]
            valid_batch_targets = valid_target_vecs[valid_index: valid_index+batch_size]

            valid_feature_var = Variable(xp.array(valid_batch_features, dtype=xp.float32))
            valid_question_var = Variable(xp.array(valid_batch_questions, dtype=xp.float32))
            valid_target_var = Variable(xp.array(valid_batch_targets, dtype=xp.float32))

            with chainer.using_config('train', False), chainer.no_backprop_mode():
                valid_loss, valid_acc, valid_size = forward(valid_target_var, valid_feature_var, valid_question_var, CaptionNet, eos)

            valid_sum_loss += valid_loss.data.tolist()
            valid_sum_acc += valid_acc.tolist()
            valid_sum_size += valid_size.tolist()

        print("{:3d} epoch train loss : {:.5f}, acc : {:.5f} | valid loss : {:.5f}, acc : {:.5f}\n".format(epoch, sum_loss/sum_size, sum_acc/sum_size, valid_sum_loss/valid_sum_size, valid_sum_acc/valid_sum_size))

        if (epoch > 0) and (epoch % 10 == 0):
            serializers.save_hdf5(model_save_path + 'model_' + str(epoch) + '.h5', CaptionNet)
            serializers.save_hdf5(model_save_path + 'optimizer_' + str(epoch) + '.h5', optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--GPU', type=int, default=0)
    parser.add_argument('-model', '--MODEL_PATH', default='../data/')
    parser.add_argument('-word2id', '--WORD2ID_PATH', default='../data/word2id.pickle')
    parser.add_argument('-all_data', '--ALL_DATA_PATH', default='../data/processed_data.pickle')
    parser.add_argument('-all_feature', '--ALL_FEATURE_PATH', default='../data/all_feature.pickle')
    args = parser.parse_args()

    gpu_device = args.GPU

    cuda.get_device_from_id(gpu_device).use()
    xp = cuda.cupy

    main(args)
