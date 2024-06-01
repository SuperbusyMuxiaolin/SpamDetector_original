import math
import os
import re
import string
import json


class Spamdetect:

    def __init__(self):
        self.word_counts = None
        self.log_class_p = None
        self.num_emails = None
        self.all_words = None

    # 将文件名和标签分开
    def getfeature(self, directory=None):
        X = []  # 特征
        y = []  # 标签
        train_files = os.listdir(directory)
        for file in train_files:
            label = int(file.split('_')[0])  # 获取文件名中的标签（第一个值）
            with open(os.path.join(directory, file), encoding='latin-1') as f:
                X.append(f.read())
                y.append(label)
        # print(len(X))
        # print(len(y))
        # print(y.__getitem__(0))
        return X, y

    # 去除所有标点符号
    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    # 分开每个单词
    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    # 计算某个单词出现的次数
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def train(self, features, labels):
        # 存储邮件两种种类的个数
        self.num_emails = {"spam": sum(1 for label in labels if label == 1),
                           "ham": sum(1 for label in labels if label == 0)}
        with open('num_emails.json', 'w', encoding='utf-8') as f:
            json.dump(self.num_emails, f, ensure_ascii=False, indent=4)
        # 两种邮件比例的log
        self.log_class_p = {
            "spam": math.log(self.num_emails['spam'] / (self.num_emails['spam'] + self.num_emails['ham'])),
            "ham": math.log(self.num_emails['ham'] / (self.num_emails['spam'] + self.num_emails['ham']))
        }
        # 将log_class_p字典保存为 JSON 文件, 方便调用模型
        with open('log_class_p.json', 'w', encoding='utf-8') as f:
            json.dump(self.log_class_p, f, ensure_ascii=False, indent=4)

        self.all_words = []

        self.word_counts = {'spam': {}, 'ham': {}}

        k = 0
        for f, la in zip(features, labels):
            k += 1
            print("k:", k)
            if la == 1:
                c = 'spam'
            else:
                c = 'ham'
            single_words_counts = self.get_word_counts(self.tokenize(f))
            print(single_words_counts)
            for word, count in single_words_counts.items():
                if word not in self.all_words:
                    self.all_words.append(word)
                # print(word)
                # print(all_words)
                # 如果word不在字典里，先初始化以word为第二层键值的count值为0
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
                self.word_counts[c][word] += 1
                # self.word_counts[c][word] += count

        # print(self.word_counts.__getitem__('spam'))
        # print(self.log_class_p["spam"])
        # print(self.log_class_p["ham"])
        with open('all_words.json', 'w', encoding='utf-8') as f:
            json.dump(self.all_words, f, ensure_ascii=False, indent=4)
        with open('word_counts.json', 'w', encoding='utf-8') as f:
            json.dump(self.word_counts, f, ensure_ascii=False, indent=4)

    def predict(self, features_t, labels_t):
        # 读取 JSON 文件中的模型数据
        with open('num_emails.json', 'r', encoding='utf-8') as f:
            num_emails = json.load(f)

        with open('log_class_p.json', 'r', encoding='utf-8') as f:
            log_class_p = json.load(f)
        # print(log_class_p)
        with open('word_counts.json', 'r', encoding='utf-8') as f:
            word_counts = json.load(f)
        # print(word_counts)
        with open('all_words.json', 'r', encoding='utf-8') as f:
            all_words = json.load(f)
        y_predict = []
        k = 0
        for feature in features_t:
            k += 1
            single_word_counts = self.get_word_counts(self.tokenize(feature))
            spam = log_class_p['spam']
            ham = log_class_p['ham']
            for word, count in single_word_counts.items():
                if word not in all_words:
                    continue
                else:
                    if word in word_counts['spam'].keys() and word in word_counts['ham'].keys():
                        # 词集模型,每个词在同一篇邮件里只考虑其是否出现，不考虑同一篇邮件里的词频
                        # log_word_given_spam = math.log(
                        #     (word_counts['spam'][word] + 1) / (num_emails['spam'] + len(all_words)))
                        # log_word_given_ham = math.log(
                        #     (word_counts['ham'][word] + 1) / (num_emails['ham'] + len(all_words)))
                        # 词袋模型，每个词还考虑了词在一篇邮件里出现的频率
                        log_word_given_spam = math.log(
                            (word_counts['spam'][word] + 1) / (sum(word_counts['spam'].values()) + len(all_words)))
                        log_word_given_ham = math.log(
                            (word_counts['ham'][word] + 1) / (sum(word_counts['ham'].values()) + len(all_words)))
                    if word in word_counts['spam'].keys() and word not in word_counts['ham'].keys():
                        # 词集模型,每个词在同一篇邮件里只考虑其是否出现，不考虑同一篇邮件里的词频
                        # log_word_given_spam = math.log(
                        #     (word_counts['spam'][word] + 1) / (num_emails['spam'] + len(all_words)))
                        # log_word_given_ham = math.log(
                        #     1 / (num_emails['ham'] + len(all_words)))
                        # 词袋模型，每个词还考虑了词在一篇邮件里出现的频率
                        log_word_given_spam = math.log(
                           (word_counts['spam'][word] + 1) / (sum(word_counts['spam'].values()) + len(all_words)))
                        log_word_given_ham = math.log(
                           1 / (sum(word_counts['ham'].values()) + len(all_words)))
                    if word not in word_counts['spam'].keys() and word not in word_counts['ham'].keys():
                        # 词集模型,每个词在同一篇邮件里只考虑其是否出现，不考虑同一篇邮件里的词频
                        log_word_given_spam = math.log(
                            1 / (num_emails['spam'] + len(all_words)))
                        log_word_given_ham = math.log(
                            (word_counts['ham'][word] + 1) / (num_emails['ham'] + len(all_words)))
                        # 词袋模型，每个词还考虑了词在一篇邮件里出现的频率
                        log_word_given_spam = math.log(
                           1 / (sum(word_counts['spam'].values()) + len(all_words)))
                        log_word_given_ham = math.log(
                           (word_counts['ham'][word] + 1) / (sum(word_counts['ham'].values()) + len(all_words)))
                spam += log_word_given_spam
                ham += log_word_given_ham
            if spam > ham:
                y_predict.append(1)
            else:
                y_predict.append(0)
        wright = 0
        all_test=0
        for y_p,l_t in zip(y_predict,labels_t):
            all_test +=1
            if y_p==l_t:
                wright+=1
        accuracy=wright/all_test
        print(accuracy)
