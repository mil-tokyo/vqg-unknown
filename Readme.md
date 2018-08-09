## Visual Question Generation for Class Acquisition of Unknown Objects

This is the implementation of [Visual Question Generation for Class Acquisition of Unknown Objects](https://arxiv.org/abs/1808.01821) by Kohei Uehara, Antonio Tejero-De-Pablos, Yoshitaka Ushiku and Tatsuya Harada (ECCV 2018).

### Requirements

- Python 3.6
- Chainer 3.2.0
- cupy 2.2.0
- matplotlib 2.1.0
- scikit-image
- [selectivesearch](https://github.com/AlpacaDB/selectivesearch)
- [pyimagesaliency](https://github.com/yhenon/pyimgsaliency)
    - This code is for python2, so you need to add option ```use_2to3 = True``` to ```setup.py```.
- opencv-python
- nltk
- numpy

### Usage

#### Download

You can download files for this code, and if you want to use them, put them in the ```/data``` folder.

- pretrained ResNet model [[Download](https://drive.google.com/open?id=1hTtzNhiwzLB2nQhWlNzxdW1hXp1XFyN4)]
- pretrained VQG model  [[Download](https://drive.google.com/open?id=1n_LMkKMPH3OSxQB3nsT-EpRhDwmfUePm)]
- our Dataset  [[Download](https://drive.google.com/open?id=1ihhoSgW-hZIIVLPSp6g__EHQkfDqP4VQ)]
    - This contains questions, answers, and question target for each of the images from Visual Genome dataset.
- word embeddings  [[Download](https://drive.google.com/open?id=1m3uKAAlqTG9YhwQ8qR2hOtgzeX3IKmMb)]
    - This contains word vectors by poincare embeddings for each of the target words from wordnet synset.
- word id mappings  [[Download](https://drive.google.com/open?id=1mYrRcV3k48o3gMNTim5htD6btjvMKA-E)]

Also, you need to download Visual Genome images, and put them to ```/data/images/```


#### Test

You can test our entire module on your image (generate a question for an unknown object in an image), run
```
python test.py -image path/to/image
```

#### Train

First, preprocess the data by

```
python src/preprocess.py
```

Next, extract features (you can download extracted features [here](https://drive.google.com/open?id=1j3re6rCKD6OfodoelYrLzRrBDGoxq9rC))
```
python src/feature_extract.py
```

Then, for training visual question generation module, run
```
python src/train.py
```

Before run this code, you need to put data to ```/data``` folder correctly.

If you want to test our question generation module (generate questions from image, target region, and target word), run
```
python q_test.py
```
then you can get questions (q_result.json) for each target in images.
