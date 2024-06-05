import math
import os
import re
import string
import json


class Spamdetect:

    def __init__(self):
        self.word_counts = {'spam': {}, 'ham': {}}
        self.log_class_p = None
        self.num_emails = None
        self.all_words = []
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
                           "your", "yours", "yourself", "yourselves", "subject"}

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
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

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
            print("k:", k)
            if la == 1:
                c = 'spam'
            else:
                c = 'ham'
            single_words_counts = self.get_word_counts(self.tokenize(f))
            # print(single_words_counts)
            for word, count in single_words_counts.items():
                # 如果单词不再all_words中，则将单词加入all_words中
                if word not in self.all_words:
                    self.all_words.append(word)
                # 如果word不在字典里，先初始化以word为第二层键值的count值为0
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
                # self.word_counts[c][word] += 1
                self.word_counts[c][word] += count

        # 将参数保存为json文件, 方便调用模型
        with open('num_emails.json', 'w', encoding='utf-8') as f:
            json.dump(self.num_emails, f, ensure_ascii=False, indent=4)
        with open('log_class_p.json', 'w', encoding='utf-8') as f:
            json.dump(self.log_class_p, f, ensure_ascii=False, indent=4)
        with open('all_words.json', 'w', encoding='utf-8') as f:
            json.dump(self.all_words, f, ensure_ascii=False, indent=4)
        with open('word_counts.json', 'w', encoding='utf-8') as f:
            json.dump(self.word_counts, f, ensure_ascii=False, indent=4)

    def predict(self, features_t, labels_t):
        # 读取 JSON 文件中的模型数据
        with open('num_emails.json', 'r', encoding='utf-8') as f:
            num_emails = json.load(f)
            print("spam:", num_emails["spam"])
            print("ham:", num_emails["ham"])
        with open('log_class_p.json', 'r', encoding='utf-8') as f:
            log_class_p = json.load(f)
        with open('word_counts.json', 'r', encoding='utf-8') as f:
            word_counts = json.load(f)
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
                word_in_spam = word_counts['spam'].get(word, 0.0)
                word_in_ham = word_counts['ham'].get(word, 0.0)
                # 还考虑了单词在单篇文章中出现的次数
                log_word_given_spam = math.log(
                    (word_in_spam + 1) / (sum(word_counts['spam'].values()) + len(all_words)))
                log_word_given_ham = math.log((word_in_ham + 1) / (sum(word_counts['ham'].values()) + len(all_words)))

                # 仅仅考虑单词在单篇文章中是否出现
                # log_word_given_spam = math.log(
                #     (word_in_spam + 1) / (num_emails['spam'] + len(all_words)))
                # log_word_given_ham = math.log((word_in_ham + 1) / (num_emails['spam'] + len(all_words)))
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
        print(accuracy)
