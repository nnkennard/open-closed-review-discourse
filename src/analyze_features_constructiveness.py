import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn import preprocessing

corr_function = pearsonr
LAST_ANN_COL = 13
LAST_GEN_FEAT_COL = 55

def load_features():
    file_path = '../data/features/common_ann.csv'
    df = pd.read_csv(file_path)
    res_df = df.dropna()
    liwc_df = pd.concat([res_df.iloc[:, :LAST_ANN_COL], res_df.iloc[:, LAST_GEN_FEAT_COL:]], axis=1)
    gen_df = res_df.iloc[:, :LAST_GEN_FEAT_COL]

    liwc_df.to_csv('../data/features/normalized_ann_liwc.csv', index=False)
    gen_df.to_csv('../data/features/normalized_ann_gen.csv', index=False)
    return gen_df


def get_significant_features(df):
    annotation_types = ['constructiveness']
    features = df.columns[LAST_ANN_COL:]
    num_output = len(annotation_types)
    num_features = len(features)
    significant_features = []
    print("Number of features           : ", num_features)
    print("Bonferonni corrected p-value : ", (0.05 / num_features))
    p_values = np.ones((num_features, num_output))
    c_values = np.ones((num_features, num_output))

    significant_features = set()
    for i, feature in enumerate(features):
        for j, category in enumerate(annotation_types):
            y = df[category].values
            x = df[feature].values
            corr, p_val = corr_function(x, y)
            p_values[i, j] = p_val
            c_values[i, j] = corr
            if p_val < (0.05 / num_features):
                significant_features.add(i)
    return list(significant_features), p_values, c_values


def main():
    df = load_features()
    features = df.columns[LAST_ANN_COL:]
    significant_features, p, c = get_significant_features(df)
    
    print('-'*80)
    print("Significant Feature", "\t->", "P-value", "\t\t", "Correlation")
    print('-'*80)
    with open('../data/features/all_significant.csv', 'w+') as f:
        for feature_idx in significant_features:
            feature = features[feature_idx]
            f.write(feature + '\n')
            print(feature, "\t->", p[feature_idx, 0], "\t", c[feature_idx, 0])


if __name__ == "__main__":
    main()