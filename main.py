from spamdetect import Spamdetect
import time

spamdetect = Spamdetect()
X, y = spamdetect.getfeature("processed_train_data")
spamdetect.train(X, y)
spamdetect.train2(X, y)  # 获得关键词语

X2, y2 = spamdetect.getfeature("processed_test_data")

start_time_predict1 = time.time()
spamdetect.predict1(X2, y2)
end_time_predict1 = time.time()
execution_time_predict1 = end_time_predict1 - start_time_predict1

start_time_predict2 = time.time()
spamdetect.predict2(X2, y2)
end_time_predict2 = time.time()
execution_time_predict2 = end_time_predict2 - start_time_predict2

start_time_predict3 = time.time()
spamdetect.predict3(X2, y2)
end_time_predict3 = time.time()
execution_time_predict3 = end_time_predict3 - start_time_predict3

start_time_predict4 = time.time()
spamdetect.predict4(X2, y2)
end_time_predict4 = time.time()
execution_time_predict4 = end_time_predict4 - start_time_predict4

print(f"Execution time of predict1: {execution_time_predict1} seconds")
print(f"Execution time of predict3: {execution_time_predict3} seconds")
print(f"Execution time of predict4: {execution_time_predict4} seconds")
