import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def text_preprocess(ds):
    for i in range(len(ds)):
        main_words = re.sub(r'".*"', '<QUOTE>', ds[i])
        main_words = re.sub(r"'.*'", '<QUOTE>', main_words)
        main_words = re.sub(r'“.*“', '<QUOTE>', main_words)
        main_words = (main_words.lower()).split()
        main_words = [w for w in main_words if not w in set(stopwords.words('english'))]  # Remove stopwords

        lem = WordNetLemmatizer()
        main_words = [lem.lemmatize(w) for w in main_words if len(w) > 1]  # Group different forms of the same word

        main_words = ' '.join(main_words)
        ds[i] = main_words

    return ds

def grid_search_cv(X_train, y_train):
    # Grid search and cv for MLP
    param_grid1 = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
                   'learning_rate_init': [0.1, 0.01, 0.001, 0.0001]}
    cls1 = MLPClassifier()
    grid = GridSearchCV(cls1, param_grid1, scoring='accuracy', cv=5)
    grid.fit(X_train, y_train)
    print('The best parameters are for MLPClassifier: ')
    print(grid.best_params_)
    print('The best accuracy for MLPClassifier: ')
    print(grid.best_score_)

    # Grid search and cv for SVM
    param_grid2 = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    cls2 = SVC()
    grid = GridSearchCV(cls2, param_grid2, scoring='accuracy', cv=5)
    grid.fit(X_train, y_train)
    print('The best parameters are for SVM: ')
    print(grid.best_params_)
    print('The best accuracy for SVM: ')
    print(grid.best_score_)


def baseline(X_train, X_test, y_train, y_test):
    # Grid search and cross validation
    # grid_search_cv(X_train, y_train)

    # Test MLPClassifier the model with the best parameters
    # mlp = MLPClassifier(activation='logistic', learning_rate_init=0.001, solver='lbfgs') # quote is much better in this model, but overall is worse
    mlp = MLPClassifier(activation='logistic', learning_rate_init=0.0001, solver='adam')
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    report_mlp = classification_report(y_test, y_pred)
    print('\n Multi-layer Perceptron classifier')
    print('\n F1-score: ', f1_score(y_test, y_pred, average='micro'))
    print('\n Classification Report')
    print('======================================================')
    print('\n', report_mlp)

    joblib.dump(mlp, 'models/mlp_q.pkl')

    # Test SVM the model with the best parameters
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    report_svm = classification_report(y_test, y_pred)
    print('\n SVM')
    print('\n F1-score: ', f1_score(y_test, y_pred, average='micro'))
    print('\n Classification Report')
    print('======================================================')
    print('\n', report_svm)

    joblib.dump(svm, 'models/svm.pkl')

def get_baseline_models():
    data_df = pd.read_csv('dataset.csv')
    data_df['processed_reviews'] = text_preprocess(data_df.reviews.values)

    # The training and test datadet for baseline
    td = TfidfVectorizer(max_features=4500)
    X_train = td.fit_transform(data_df[data_df.data_type == 'train'].processed_reviews.values).toarray()
    X_test = td.transform(data_df[data_df.data_type == 'test'].processed_reviews.values).toarray()
    y_train = data_df[data_df.data_type == 'train'].label.values
    y_test = data_df[data_df.data_type == 'test'].label.values

    # Train and test baseline models
    baseline(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    get_baseline_models()