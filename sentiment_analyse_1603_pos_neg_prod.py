import subprocess
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import tflearn
import nltk
from collections import Counter
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer
from tflearn.data_utils import to_categorical

tokenizer = TweetTokenizer()
stopwords = nltk.corpus.stopwords.words('russian')
stemer = RussianStemmer()
re_russian = re.compile('[^а-яА-Я ]')
stem_cache = {}
len_voc = 6500


def read_positive_data(path_to_file):  
    """"Считывает датасет с позитивными постами"""
    positive = pd.read_csv(path_to_file)
    positive.fillna('', inplace=True)
    return positive


def read_negative_data(path_to_file):  
    """"Считывает датасет с негативными постами"""
    negative = pd.read_csv(path_to_file)
    negative.fillna('', inplace=True)
    return negative


def get_stem(token):  
    """"
    1.Обрезает окончание слова,
    2.Приводит весь текст к нижнему регистру
    3.Убирает латиницу
    :param token: входящee слово
    :return stem: обрезанное слово без латиницы
     """
    stem = stem_cache.get(token)
    if stem:
        return stem
    token = re_russian.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem


def count_unique_tokens_in_text(dataset):
    """"Создает словарь вида  - стемма слова:кол-во упоминаний
    :param dataset: датасет с текстом
    заполняет словарь stem_count(глобал.переменная) с кол-вом упоминаний каждой стеммы
     """
    for _, data in dataset.iterrows():
        text = data[0]  # Здесь указывается номер колонки в датасете ,в которой содержится текст
        tokens = tokenizer.tokenize(text)
        for t in tokens[:40]:
            stem = get_stem(t)
            stem_count[stem] += 1


def senten_to_vector(post):  
    """"Создает предложение длиной 6500 символов из 0 и 1, где 1-ка ставится если присутствует слово из словаря
    :param post: текст
    :return vector: словарь с кол-вом упоминаний каждого слова
     """
    vector = np.zeros(len_voc, dtype=np.int_)
    for word in tokenizer.tokenize(post):
        stem = get_stem(word)
        idx = token_2_idx.get(stem)
        if idx is not None:
            vector[idx] = 1
    return vector


def build_model(learning_rate):  
    """Создает нейросеть описанной конфигурации
    :param learning_rate: шаг обучения
    :return model: нейросеть
    """
    tf.Graph()
    net = tflearn.input_data([None, len_voc])
    net = tflearn.fully_connected(net, 256, activation='PReLU')
    net = tflearn.fully_connected(net, 64, activation='Tanh')
    net = tflearn.fully_connected(net, 2, activation='sigmoid')
    regression = tflearn.regression(
        net,
        optimizer='sgd',
        learning_rate=learning_rate,
        loss='categorical_crossentropy')
    model = tflearn.DNN(regression)
    return model


def train_model(X, y):
    """Обучает модель
    :param X: вектор с входящими данными(слова в виде числового вектора)
    :param y: массив с метками : 1-позитив,0-негатив(массив вида[[1,0],[1,0]..])
    :return model: обученная модель
    """
    model.fit(X, y, validation_set=0.2, show_metric=True, batch_size=64, n_epoch=20)
    return model


def test_post(post, model):  # используется ПОСЛЕ ТОГО, как мы обучили модель и хотим разметить НОВЫЕ посты
    """Определяет вероятность отнесения поста к позитивному(1) или негативному(0)
    :param post: входящий текст
    :param model: обученная модель
    :return model.predict: вероятность отнесения к позитиву"""
    post_vector = senten_to_vector(post)
    return round(model.predict([post_vector])[0][1], 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ping script')
    parser.add_argument('--negative_file')
    parser.add_argument('--positive_file')
    args = parser.parse_args()

    positive = read_positive_data(args.positive_file)
    negative = read_negative_data(args.negative_file)


    """Создаем словарь из слов в обоих датасетах"""
    stem_count = Counter()  # инициализируем словарь-счетчик
    count_unique_tokens_in_text(positive)
    count_unique_tokens_in_text(negative)

    """Сортируем словарь по убыванию кол-ва упоминаний слов
    убираем предлоги и стоп-слова,кроме частицы 'не',
    ограничиваем его длиной в 6500 слов"""
    vocab = ['не']
    for stem_w in sorted(stem_count, key=stem_count.get, reverse=True):
        if len(stem_w) > 2 and stem_w not in stopwords:
            vocab.append(stem_w)
    token_2_idx = {vocab[i]: i for i in range(len_voc)}

    """Создаем из постов векторы  из 0 и 1 для подачи их в модель в качестве Х"""
    data_vectors = np.zeros((len(positive) + len(negative), len_voc), dtype=np.int_)
    for i, post in positive.iterrows():
        data_vectors[i] = senten_to_vector(post[0])
    for i, post in negative.iterrows():
        data_vectors[i + len(positive)] = senten_to_vector(post[0])
    """Создаем из датасетов векторы из 0 и 1 для подачи их в модель в качестве Y"""
    labels = np.append(
        np.ones(len(positive), dtype=np.int_),
        np.zeros(len(negative), dtype=np.int_))

    X = data_vectors
    y = to_categorical(labels, 2)
    model = build_model(0.3)  # создаем модель
    train_model(X, y)  # обучаем модель
