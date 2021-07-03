import pandas as pd
import glob
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split



def import_dataset():
    data_df = pd.DataFrame(columns=['reviews', 'type'])
    rpath = 'dataset/iclr_anno_final/'
    for file in glob.glob(rpath + "*.txt"):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                for i in range(len(line)):
                    if line[i].isspace() == True:
                        data_df.loc[len(data_df)] = [line[i + 1:].strip(), line[: i]]
                        break

    possible_labels = data_df.type.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    data_df['label'] = data_df.type.replace(label_dict)
    # {'fact': 0, 'evaluation': 1, 'request': 2, 'reference': 3, 'non-arg': 4, 'quote': 5}
    print(label_dict)

    return data_df

def split_dataset(df):
    X_train, X_test, y_train, y_test = train_test_split(df.index.values,
                                                      df.label.values,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=df.label.values)
    df['data_type'] = ['not_set'] * df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_test, 'data_type'] = 'test'

    return df

def get_dataset():
    # Import and preprocessing the dataset
    data_df = import_dataset()
    data_df = split_dataset(data_df)
    data_df.to_csv('dataset.csv')
    print(data_df.head())

if __name__ == '__main__':
    get_dataset()
