import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


LAST_ANN_COL = 13
LAST_GEN_FEAT_COL = 55

def feature_importance(file_path):
    # Download and preprocess data
    df = pd.read_csv(file_path)

    X = df.iloc[:, LAST_ANN_COL:]
    y = df.iloc[:, 10].values

    # Desicion Tree feature importance
    model = DecisionTreeRegressor()
    model.fit(X, y)
    importance = model.feature_importances_

    cols = list(X.columns)
    print('Feature importance for Decision Tree')
    print_results(importance, cols)
    result = pd.DataFrame(columns=cols)
    result.loc[0] = importance

    # Random Forest Tree feature importance
    model = RandomForestRegressor()
    model.fit(X, y)
    importance = model.feature_importances_

    print('Feature importance for Random Forest')
    print_results(importance, cols)
    result.loc[1] = importance
    result.to_csv('feature_importance/feature_importance_gen.csv')

def print_results(importance, cols):
    results = []
    for i, v in enumerate(importance):
        results.append((v, cols[i]))

    res = sorted(results, reverse=True)
    for i, v in enumerate(res):
        if not res[i][0]:
            continue
        print('Feature: %s, Score: %.5f' % (res[i][1], res[i][0]))

if __name__ == '__main__':
    feature_importance('../data/features/normalized_ann_gen.csv')