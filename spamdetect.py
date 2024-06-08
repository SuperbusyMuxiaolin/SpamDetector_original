import math
import os
import re
import string
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


class Spamdetect:

    def __init__(self):
        self.word_counts = {'spam': Counter(), 'ham': Counter()}
        self.log_class_p = None
        self.num_emails = None
        self.all_words = set()
        # 停用词，对分类几乎没影响
        self.stop_words = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
                           "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
                           "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
                           "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
                           "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
                           "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
                           "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
                           "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
                           "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
                           "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
                           "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
                           "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                           "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
                           "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
                           "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
                           "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
                           "your", "yours", "yourself", "yourselves", "subject", "http", "www", "com", "net", "org"}

    # 将文件名和标签分开
    def getfeature(self, directory=None):
        X = []  # 特征
        y = []  # 标签
        train_files = os.listdir(directory)
        for file in train_files:
            label = int(file.split('_')[0])  # 获取文件名中的标签（第一个值）
            with open(os.path.join(directory, file), encoding='latin-1') as f:
                X.append(f.read())  # 存储邮件内容的字符串列表
                y.append(label)  # 存储标签的int列表
        return X, y

    # 去除所有标点符号
    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    # 分开每个单词并全部设定为小写，去除停用词
    def tokenize(self, text):
        text = self.clean(text).lower()
        words = re.split("\W+", text)
        return [word for word in words if word not in self.stop_words and word != ""]

    # 计算words列表中每个单词出现的频率
    def get_word_counts(self, words, flag):
        if flag == 1:
            # 返回一个字典，每个单词对应出现次数
            return Counter(words)
        else:
            # 返回一个字典，每个单词对应1
            return {word: 1 for word in set(words)}

    def train(self, features, labels):
        # 存储邮件两种种类的个数
        self.num_emails = {"spam": sum(1 for label in labels if label == 1),
                           "ham": sum(1 for label in labels if label == 0)}

        # 存储两种邮件比例的log
        self.log_class_p = {
            "spam": math.log(self.num_emails['spam'] / (self.num_emails['spam'] + self.num_emails['ham'])),
            "ham": math.log(self.num_emails['ham'] / (self.num_emails['spam'] + self.num_emails['ham']))
        }

        # 统计两种邮件各自单词词频以及所有出现的单词
        k = 0
        for f, la in zip(features, labels):
            k += 1
            print("train_k:", k)
            if la == 1:
                c = 'spam'
            else:
                c = 'ham'
            words = self.tokenize(f)  # 分词

            # flag=1时考虑词频，为方法3;flag=0时不考虑词频，为方法1 运行predict2要求flag为0
            single_word_counts = self.get_word_counts(words, 0)
            self.word_counts[c].update(single_word_counts)
            self.all_words.update(words)

        # 将参数保存为json文件, 方便调用模型
        # save the model
        with open('model.json', 'w') as f:
            json.dump({
                'word_counts': {k: dict(v) for k, v in self.word_counts.items()},
                'log_class_p': self.log_class_p,
                'num_emails': self.num_emails,
                'all_words': list(self.all_words)
            }, f)

    def train2(self, features, labels):
        # Separate features based on their labels
        features_1 = " ".join([f for f, l in zip(features, labels) if l == 1])
        features_0 = " ".join([f for f, l in zip(features, labels) if l == 0])

        # Combine into a list
        features_list = [features_1, features_0]

        # Apply TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        vectorizer.fit_transform(features_list)
        feature_names = vectorizer.get_feature_names_out()
        #print(feature_names)

        # Save feature_names to a JSON file
        with open('feature_names.json', 'w') as f:
            json.dump(feature_names.tolist(), f)

    # 方法一 or 方法三
    def predict1(self, features_t, labels_t):
        # 读取 JSON 文件中的模型数据
        with open('model.json', 'r') as f:
            model = json.load(f)
        word_counts = {k: Counter(v) for k, v in model['word_counts'].items()}
        log_class_p = model['log_class_p']
        num_emails = model['num_emails']
        all_words = set(model['all_words'])
        dum = sum(word_counts['spam'].values()) + len(all_words)
        y_predict = []
        k = 0
        for feature in features_t:
            k += 1
            print("predict1_k:", k)
            single_word_counts = self.get_word_counts(self.tokenize(feature), 0)  # 获取单个邮件的词和词频
            spam = log_class_p['spam']
            ham = log_class_p['ham']
            for word in all_words:
                word_in_spam = word_counts['spam'].get(word, 0.0)
                word_in_ham = word_counts['ham'].get(word, 0.0)
                if word in single_word_counts:
                    # # 计算条件概率时考虑了单词在单篇文章中出现的次数
                    # log_spam = math.log((word_in_spam + 1) / dum)
                    #
                    # log_ham = math.log((word_in_ham + 1) / dum)
                    # 计算条件概率时仅仅考虑单词在单篇文章中是否出现,不考虑频次
                    log_spam = math.log(
                        (word_in_spam + 1) / (num_emails['spam'] + 2))
                    log_ham = math.log(
                        (word_in_ham + 1) / (num_emails['ham'] + 2))

                else:
                    # 计算条件概率时考虑了单词在单篇文章中出现的次数
                    # log_spam = math.log(1 - ((word_in_spam + 1) / dum))
                    # log_ham = math.log(1 - ((word_in_ham + 1) / dum))
                    # 计算条件概率时仅仅考虑单词在单篇文章中是否出现,不考虑频次
                    log_spam = math.log(1 -
                                        ((word_in_spam + 1) / (num_emails['spam'] + 2)))
                    log_ham = math.log(1 -
                                       ((word_in_ham + 1) / (num_emails['ham'] + 2)))
                spam += log_spam
                ham += log_ham
            if spam > ham:
                y_predict.append(1)
            else:
                y_predict.append(0)

        wright = 0
        all_test = 0
        for y_p, l_t in zip(y_predict, labels_t):
            all_test += 1
            if y_p == l_t:
                wright += 1
        accuracy = wright / all_test
        print("predict1：", accuracy)

    # 方法二
    def predict2(self, features_t, labels_t):
        # 读取 JSON 文件中的模型数据
        with open('model.json', 'r') as f:
            model = json.load(f)
        word_counts = {k: Counter(v) for k, v in model['word_counts'].items()}
        log_class_p = model['log_class_p']
        num_emails = model['num_emails']
        all_words = set(model['all_words'])
        dum = sum(word_counts['spam'].values()) + len(all_words)
        y_predict = []
        k = 0
        for feature in features_t:
            k += 1
            print("predict2_k:", k)
            single_word_counts = self.get_word_counts(self.tokenize(feature), 1)
            spam = log_class_p['spam']
            ham = log_class_p['ham']
            for word in all_words:
                word_in_spam = word_counts['spam'].get(word, 0.0)
                word_in_ham = word_counts['ham'].get(word, 0.0)
                if word in single_word_counts:
                    # 计算条件概率时考虑了单词在单篇文章中出现的次数
                    # log_spam = math.log((word_in_spam + 1) / dum)
                    # log_ham = math.log((word_in_ham + 1) / dum)
                    # 计算条件概率时仅仅考虑单词在单篇文章中是否出现,不考虑频次，且按单词在文本中出现的频次进行了加权
                    log_spam = single_word_counts[word] * math.log(
                        (word_in_spam + 1) / (num_emails['spam'] + 2))
                    log_ham = single_word_counts[word] * math.log(
                        (word_in_ham + 1) / (num_emails['ham'] + 2))

                else:
                    # 计算条件概率时考虑了单词在单篇文章中出现的次数
                    # log_spam = math.log(1 - ((word_in_spam + 1) / dum))
                    # log_ham = math.log(1 - ((word_in_ham + 1) / dum))
                    # 计算条件概率时仅仅考虑单词在单篇文章中是否出现,不考虑频次
                    log_spam = math.log(1 -
                                        ((word_in_spam + 1) / (num_emails['spam'] + 2)))
                    log_ham = math.log(1 -
                                       ((word_in_ham + 1) / (num_emails['ham'] + 2)))
                spam += log_spam
                ham += log_ham
            if spam > ham:
                y_predict.append(1)
            else:
                y_predict.append(0)

        wright = 0
        all_test = 0
        for y_p, l_t in zip(y_predict, labels_t):
            all_test += 1
            if y_p == l_t:
                wright += 1
        accuracy = wright / all_test
        print("predict2", accuracy)

    # 方法四
    def predict3(self, features_t, labels_t):
        # 读取 JSON 文件中的模型数据
        with open('model.json', 'r') as f:
            model = json.load(f)
        word_counts = {k: Counter(v) for k, v in model['word_counts'].items()}
        log_class_p = model['log_class_p']
        num_emails = model['num_emails']
        all_words = set(model['all_words'])
        dum = sum(word_counts['spam'].values()) + len(all_words)
        y_predict = []
        k = 0
        for feature in features_t:
            k += 1
            print("predict3_k:", k)
            single_word_counts = self.get_word_counts(self.tokenize(feature), 0)
            spam = log_class_p['spam']
            ham = log_class_p['ham']
            for word, count in single_word_counts.items():
                if word not in all_words:
                    continue
                word_in_spam = word_counts['spam'].get(word, 0.0)
                word_in_ham = word_counts['ham'].get(word, 0.0)
                # 计算条件概率时考虑了单词在单篇文章中出现的次数
                # log_word_given_spam = math.log((word_in_spam + 1) / dum)
                # log_word_given_ham = math.log((word_in_ham + 1) / dum)

                # 计算条件概率时仅仅考虑单词在单篇文章中是否出现,不考虑频次
                log_word_given_spam = math.log(
                    (word_in_spam + 1) / (num_emails['spam'] + 2))
                log_word_given_ham = math.log(
                    (word_in_ham + 1) / (num_emails['ham'] + 2))
                spam += log_word_given_spam
                ham += log_word_given_ham
            if spam > ham:
                y_predict.append(1)
            else:
                y_predict.append(0)
        wright = 0
        all_test = 0
        for y_p, l_t in zip(y_predict, labels_t):
            all_test += 1
            if y_p == l_t:
                wright += 1
        accuracy = wright / all_test
        print("predict3:", accuracy)

    # 对应方法五
    def predict4(self, features_t, labels_t):
        # 读取 JSON 文件中的模型数据
        with open('model.json', 'r') as f:
            model = json.load(f)
        word_counts = {k: Counter(v) for k, v in model['word_counts'].items()}
        log_class_p = model['log_class_p']
        num_emails = model['num_emails']
        # Load feature_names from a JSON file
        with open('feature_names.json', 'r') as f:
            featurenames = json.load(f)
        feature_names = set(featurenames)
        y_predict = []
        k = 0
        for feature in features_t:
            k += 1
            print("predict4_k:", k)
            single_word_counts = self.get_word_counts(self.tokenize(feature), 0)  # 获取单个邮件的词和词频
            spam = log_class_p['spam']
            ham = log_class_p['ham']
            for word in feature_names:
                word_in_spam = word_counts['spam'].get(word, 0.0)
                word_in_ham = word_counts['ham'].get(word, 0.0)
                if word in single_word_counts:
                    log_spam = math.log(
                        (word_in_spam + 1) / (num_emails['spam'] + 2))
                    log_ham = math.log(
                        (word_in_ham + 1) / (num_emails['ham'] + 2))
                else:
                    log_spam = math.log(1 -
                                        ((word_in_spam + 1) / (num_emails['spam'] + 2)))
                    log_ham = math.log(1 -
                                       ((word_in_ham + 1) / (num_emails['ham'] + 2)))
                spam += log_spam
                ham += log_ham
            if spam > ham:
                y_predict.append(1)
            else:
                y_predict.append(0)

        wright = 0
        all_test = 0
        for y_p, l_t in zip(y_predict, labels_t):
            all_test += 1
            if y_p == l_t:
                wright += 1
        accuracy = wright / all_test
        print("predict4：", accuracy)
