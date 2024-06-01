import os
import random
import shutil


def get_files(directory, label, train_ratio):
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]  # 获取指定目录中所有以“.txt”结尾的文件
    random.shuffle(files)  # 随机排序
    train_size = int(len(files) * train_ratio)
    train_files = files[:train_size]
    test_files = files[train_size: len(files)]
    labeled_train_files = [(os.path.join(directory, f), label) for f in train_files]
    labeled_test_files = [(os.path.join(directory, f), label) for f in test_files]
    return labeled_train_files, labeled_test_files


def copy_and_label_files(file_list, dest_dir):
    for i, (file_path, label) in enumerate(file_list):
        dest_filename = f"{label}_{i}.txt"
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy(file_path, dest_path)


# 定义SPAM和HAM文件夹路径
spam_dir = 'original_data/enron2/spam'
ham_dir = 'original_data/enron2/ham'
dest_dir1 = 'processed_train_data'
dest_dir2 = 'processed_test_data'

# 创建目标目录（如果不存在）
os.makedirs(dest_dir1, exist_ok=True)
os.makedirs(dest_dir2, exist_ok=True)


# 获取SPAM文件（70%），标记为1
train_spam_files, test_spam_files = get_files(spam_dir, label=1, train_ratio=0.7)

# 获取HAM文件（70%），标记为0
train_ham_files, test_ham_files = get_files(ham_dir, label=0, train_ratio=0.7)

# 合并文件列表
train_files = train_spam_files + train_ham_files
test_files = test_spam_files + test_ham_files

# 将文件复制并标记到目标目录
copy_and_label_files(train_files, dest_dir1)
copy_and_label_files(test_files, dest_dir2)

print(f"Total train_files: {len(train_files)}")
print(f"Total test_files: {len(test_files)}")
