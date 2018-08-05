import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links.model.vision import resnet


class ResNet(chainer.Chain):
    def __init__(self, path, layer):
        super(ResNet, self).__init__()
        with self.init_scope():
            self.resnet = resnet.ResNetLayers(path, layer)

    def __call__(self, x):
        feature = self.resnet(x, layers=['fc6'])['fc6']
        return feature


class ImageCaption(chainer.Chain):
    dropout_ratio = 0.5

    def __init__(self, word_num, attr_num, feature_num, hidden_num):
        super(ImageCaption, self).__init__()
        with self.init_scope():
            self.word_vec = L.EmbedID(word_num, hidden_num)
            self.image_vec = L.Linear(feature_num, hidden_num)
            self.target_vec = L.Linear(attr_num, hidden_num)
            self.lstm = L.LSTM(hidden_num, hidden_num)
            self.out_word = L.Linear(hidden_num, word_num)

    def __call__(self, word, target):
        h_w = self.word_vec(word)
        h_t = self.target_vec(target)
        h = h_w + h_t
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        return self.out_word(F.dropout(h, ratio=self.dropout_ratio))

    def image_init(self, image_feature):
        self.lstm.reset_state()
        h = self.image_vec(F.dropout(image_feature, ratio=self.dropout_ratio))
        self.lstm(F.dropout(h, ratio=self.dropout_ratio))
