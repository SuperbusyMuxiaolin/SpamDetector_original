from spamdetect import Spamdetect

spamdetect = Spamdetect()
X, y = spamdetect.getfeature("processed_train_data")
spamdetect.train(X, y)
X2, y2 = spamdetect.getfeature("processed_test_data")
spamdetect.predict(X2, y2)
