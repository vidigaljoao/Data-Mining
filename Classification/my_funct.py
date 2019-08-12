import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.externals import joblib

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)



def report(model, X, y, perc_selec=None, GS=False, sel=False):
    """
    Report the model scores such as accuracy, precision and recall but also shows the confusion matrix based on stratified 10 fold cross-validation.
    
    Input: model: model(eg. Random forest),
           X: X data
           Y: Y data
           perc_selec: percentage of feature selection with F-test for feature scoring(none, 10, 20, ...)
           GS: If using grid search , True. if not , None.
           sel: If using selection feature, True. If not, None.
           
    Output: model scores such as accuracy, precision and recall but also shows the confusion matrix based on stratified 10 fold cross-validation.
    
    by João Vidigal june 2019             
    
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    
    list_acc = []
    list_prec = []
    list_recal = []
    cm_list = []
    count = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if sel==True:
            selector = SelectPercentile(f_classif, percentile=perc_selec).fit(X_train, y_train)
            model.fit(selector.transform(X_train), y_train)

            y_pred = model.predict(selector.transform(X_test))

            acc = accuracy_score(y_test, y_pred)
            list_acc.append(acc)

            prec = precision_score(y_test, y_pred, average='macro')
            list_prec.append(prec)

            recal = recall_score(y_test, y_pred, average='macro')
            list_recal.append(recal)
            if count == 0:
                cm = confusion_matrix(y_test, y_pred)
                count+=1
            else:
                cm_1 = confusion_matrix(y_test, y_pred)
                cm = cm + cm_1
        
        else:
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            list_acc.append(acc)

            prec = precision_score(y_test, y_pred, average='macro')
            list_prec.append(prec)

            recal = recall_score(y_test, y_pred, average='macro')
            list_recal.append(recal)
            if count == 0:
                cm = confusion_matrix(y_test, y_pred)
                count+=1
            else:
                cm_1 = confusion_matrix(y_test, y_pred)
                cm = cm + cm_1
            
    
    array_acc = np.array(list_acc)
    array_prec = np.array(list_prec)
    array_recal = np.array(list_recal)
    if GS == True:
        print("Tuned hyperparameters: ",model.best_params_)
        model = model.best_estimator_
            
        #joblib.dump(model, 'filename.joblib')
    
        

        
    print("Accuracy score: ",round(np.mean(array_acc),3),'+-', round(np.std(array_acc),3))
    print("Precision score:", round(np.mean(array_prec),3),'+-', round(np.std(array_prec),3))
    print("Recall:", round(np.mean(array_recal),3),'+-', round(np.std(array_recal),3))
    print('Confusion matrix:')
    joblib.dump(model, 'filename.joblib')

    
    plt.figure(figsize=(2,1.5))
    ax = sns.heatmap(cm, annot=True, cmap="YlGnBu")
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=['ALL', 'AML'], yticklabels=['ALL', 'AML'],
           ylabel='True label',
           xlabel='Predicted label')



    
