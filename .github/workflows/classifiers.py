from sklearn.metrics import precision_score, recall_score

def eval_confusion(y_pred, y_true=y_train):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {'precision': precision, 'recall': recall}


from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# classification models
sgd_clf = SGDClassifier(random_state=42, max_iter=100)
mlp_clf = MLPClassifier(hidden_layer_sizes=(16,))

classifiers = {
    'SGD': sgd_clf,
    'MLP': mlp_clf
}

from sklearn.model_selection import cross_val_predict

# make predictions using each model
y_preds = {}
for clf_name, clf in classifiers.items():
    y_preds[clf_name] = cross_val_predict(clf, X_train_prepared, y_train, cv=3)


for clf_name, y_pred in y_preds.items():
    conf = eval_confusion(y_pred)
    print("{}:".format(clf_name))
    print("precision: {}".format(conf['precision']))
    print("recall: {}".format(conf['recall']))
