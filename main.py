from sklearn.utils import shuffle

from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier


def test_classifier(classifier, data, target, n_samples):
    classifier.fit(data[:n_samples // 2], target[:n_samples // 2])

    expected = target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


print("Downloading dataset")
# digits = datasets.fetch_openml('mnist_784', version=1)
digits = datasets.fetch_mldata('MNIST original')

print("Flattening data")
n_samples = len(digits.data)
d = digits.data.reshape(len(digits.data), -1)
t = digits.target

data, target = shuffle(d, t, random_state=0)

print(n_samples)
print(data.shape)

clf1 = MLPClassifier(hidden_layer_sizes=100)
clf2 = DecisionTreeClassifier()
clf3 = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)

test_classifier(clf1, data, target, n_samples)
test_classifier(clf2, data, target, n_samples)
test_classifier(clf3, data, target, n_samples)
test_classifier(sclf, data, target, n_samples)
