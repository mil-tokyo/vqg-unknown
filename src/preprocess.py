import pickle
import argparse


def main(args):
    all_data_path = args.ALL_DATA_PATH
    word_emb_path = args.WORD_EMB_PATH
    word2id_path = args.WORD2ID_PATH

    with open(all_data_path, 'rb') as f:
        all_data = pickle.load(f)

    with open(word2id_path, 'rb') as f:
        word2id = pickle.load(f)

    with open(word_emb_path, 'rb') as f:
        word_embeddings = pickle.load(f)

    max_len = max([len(x['question'].split()) for x in all_data])

    for each_data in all_data:
        question = each_data['question'][:-1].split()
        q_ids = [word2id['<s>']]
        for i in range(max_len + 1):
            if i < len(question):
                each_word = question[i].lower()
                if each_word in word2id:
                    word_id = word2id[each_word]
                    q_ids.append(word_id)
                else:
                    q_ids.append(word2id['<unk>'])
            else:
                q_ids.append(word2id['</s>'])
        each_data['question_word_id'] = q_ids
        each_data['target_vec'] = word_embeddings[each_data['target']]

    with open('../data/processed_data.pickle', 'wb') as f:
        pickle.dump(all_data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-word2id', '--WORD2ID_PATH', default='../data/word2id.pickle')
    parser.add_argument('-all_data', '--ALL_DATA_PATH', default='../data/all_data.pickle')
    parser.add_argument('-word_emb', '--WORD_EMB_PATH', default='../data/word_embeddings.pickle')
    args = parser.parse_args()

    main(args)
