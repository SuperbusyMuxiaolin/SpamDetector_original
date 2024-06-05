from spamdetect import Spamdetect

spamdetect = Spamdetect()
X, y = spamdetect.getfeature("processed_train_data")
spamdetect.train(X, y)
X2, y2 = spamdetect.getfeature("processed_test_data")
spamdetect.predict1(X2, y2)
spamdetect.predict2(X2, y2)
