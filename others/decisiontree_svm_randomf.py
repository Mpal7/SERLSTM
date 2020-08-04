import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def dectree_svm_rf(X_train,y_train,X_test,y_test):

    print("\n####DECISION TREE####")
    dtree_model = DecisionTreeClassifier(max_depth=6).fit(X_train, y_train)
    dtree_predictions = dtree_model.predict(X_test)

    print(accuracy_score(y_true=y_test, y_pred=dtree_predictions))
    print(classification_report(y_test, dtree_predictions))
    # creating a confusion matrix
    print(confusion_matrix(y_test.argmax(axis=1), dtree_predictions.argmax(axis=1)))

    print("\n####SUPPORT VECTOR MACHINE####")
    y_train_svc = np.argmax(y_train,axis = 1)
    y_test_svc  = np.argmax(y_test,axis = 1)
    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train_svc)
    svm_predictions = svm_model_linear.predict(X_test)

    print(accuracy_score(y_true=y_test_svc, y_pred=svm_predictions))
    print(classification_report(y_test_svc, svm_predictions))
    # creating a confusion matrix
    print(confusion_matrix(y_test_svc, svm_predictions))

    print("\n####RANDOM FOREST####")

    classifier = RandomForestClassifier(n_estimators=100, random_state=0)


    classifier.fit(X_train, y_train)
    c_p = classifier.predict(X_test)
    print(accuracy_score(y_true=y_test, y_pred=c_p))
    print(classification_report(y_test, c_p))
    # creating a confusion matrix
    print(confusion_matrix(y_test.argmax(axis=1), c_p.argmax(axis=1)))
