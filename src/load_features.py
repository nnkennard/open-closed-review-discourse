import json
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from convokit import Corpus, Speaker, Utterance

feature_maps = {
    'iclr18': {
        'argument_col': 'ICLR_2018',
        'ps_corpus': '../data/convokit/iclr18_corpus_ps',
    },
    'neurips18': {
        'argument_col': 'NeurIPS_2018',
        'ps_corpus': '../data/convokit/neurips18_corpus_ps',
    },
    'iclr19': {
        'argument_col': 'ICLR_2019',
        'ps_corpus': '../data/convokit/iclr19_corpus_ps',
    },
    'neurips19': {
        'argument_col': 'NeurIPS_2019',
        'ps_corpus': '../data/convokit/neurips19_corpus_ps',
    }
}


def get_liwc_data(venues):
    # venues = [feature_maps[venue]['argument_col'] for venue in venues]
    file_path = '../data/liwc_data/liwc_18_19.csv'
    df = pd.read_csv(file_path, index_col=0)
    # df = df[df['venue'].isin(venues)]
    df.index = df.index.str.slice(0, -4)
    return df


def get_argument_data(venues):
    venues = [feature_maps[venue]['argument_col'] for venue in venues]
    file_path = '../data/arguments/predictions.csv'
    df = pd.read_csv(file_path, index_col=0)
    df = df[df['venue'].isin(venues)]
    argument_data = {}

    for row in df.iterrows():
        row = row[1]
        r_id = row['id']
        sent = row['review_sent']
        label = row['labels']
        if r_id not in argument_data:
            argument_data[r_id] = []
        argument_data[r_id].append([label, sent])
    # print(venues, len(argument_data))
    return argument_data


def get_aspect_data(venues):
    aspect_dir = '../data/aspect_tagger/'
    aspect_data = {}
    
    for venue in venues:
        # Load aspect tagger results
        aspect_labels = []
        with open(aspect_dir + venue + "_result.jsonl", 'r') as file:
            for line in file:
                aspect_labels.append(json.loads(line)['labels'])
        
        # Load data that maps review id to a line in the aspect results
        with open(aspect_dir + venue + "_aspect_idx.json", 'r') as f:
            index_data = json.load(f)
        
        # print(len(aspect_labels), aspect_labels[0], len(index_data))

        # Link aspects to review id
        for review_id in index_data:
            r_idx = index_data[review_id]['review']
            m_idx = index_data[review_id]['meta_review']

            aspect_data[review_id] = {
                'review_aspects' : aspect_labels[r_idx],
                'meta_review_aspects': None,
            }
            if m_idx:
                aspect_data[review_id]['meta_review_aspects'] = aspect_labels[m_idx]
    print(venues, len(aspect_data))
    return aspect_data
    

def get_all_inverse_specificity_data(venues):
    data = {}
    for venue in venues:
        file_path = '../data/specificity/{venue}/specificity.json'.format(
            venue=venue
        )
        with open(file_path, 'r') as f:
            specificity_list = json.load(f)
            venue_data = {
                s['review_id'] : [
                    1-v for v in s['specificity'] if 0 <= v <= 1
                ] for s in specificity_list
            }
            data.update(venue_data)
    return data


def get_all_specificity_data(venues):
    data = {}
    for venue in venues:
        file_path = '../data/specificity/{venue}/specificity.json'.format(
            venue=venue
        )
        with open(file_path, 'r') as f:
            specificity_list = json.load(f)
            venue_data = {k['review_id'] : k['specificity'] for k in specificity_list}
            data.update(venue_data)
    return data


def get_mean_specificity_data(venues):
    data = {}
    for venue in venues:
        file_path = '../data/specificity/{venue}/specificity_mean.csv'.format(
            venue=venue
        )
        df = pd.read_csv(file_path, index_col=0)
        data.update(df.to_dict()['specificity'])
    return data


def get_politeness_data(venues):
    corpus = Corpus(utterances=[])
    paths = [feature_maps[venue]['ps_corpus'] for venue in venues]
    for path in paths:
        venue_corpus = Corpus(path)
        # print(path)
        # venue_corpus.print_summary_stats()
        corpus = corpus.merge(venue_corpus)
    return corpus


def get_specificity_ratio(data):
    if not data:
        return 0
    num_specific = len([s for s in data if s > 0.55])
    num_total = len(data)
    return num_specific / num_total


def get_x_review_length(data):
    """
    Returns number of sentences in the review
    """
    return len(data)


def get_x_decision(data):
    return 'Accept' in data['decision']


def get_aspect_type(aspect_str):
    if aspect_str == 'summary':
        return aspect_str
    return '_'.join(aspect_str.split('_')[:-1])
        
def get_x_aspect_coverage(data):
    aspect_set = set()
    for aspect in data['review_aspects']:
        assert len(aspect) == 3
        aspect_set.add(get_aspect_type(aspect[2]))
    return len(aspect_set) / 8


def get_x_aspect_recall(data):
    aspect_set = set()
    meta_aspect_set = set()

    for aspect in data['meta_review_aspects']:
        assert len(aspect) == 3
        meta_aspect_set.add(aspect[2])
    for aspect in data['review_aspects']:
        assert len(aspect) == 3
        aspect_set.add(aspect[2])
    
    common = len(set.intersection(aspect_set, meta_aspect_set))
    meta = max(len(meta_aspect_set), 1)
    return common / meta 


def get_x_aspect_count(data, aspect_type):
    assert aspect_type is not None
    count = 0
    for aspect in data['review_aspects']:
        assert len(aspect) == 3
        r_aspect_type = get_aspect_type(aspect[2])
        if r_aspect_type == aspect_type:
            count += 1
    try:
        return count / len(data['review_aspects'])
    except ZeroDivisionError:
        return 0


def get_x_aspect_found(data, aspect_type):
    assert aspect_type is not None
    for aspect in data['review_aspects']:
        assert len(aspect) == 3
        r_aspect_type = get_aspect_type(aspect[2])
        if r_aspect_type == aspect_type:
            return 1
    return 0 


def get_x_argument_count(data, argument_type):
    assert argument_type is not None
    count = 0
    for argument in data:
        assert len(argument) == 2
        if argument[0] == argument_type:
            count += 1
    try:
        return count / len(data)
    except ZeroDivisionError:
        return 0


def get_x_argument_found(data, argument_type):
    assert argument_type is not None
    for argument in data:
        assert len(argument) == 2
        if argument[0] == argument_type:
            return 1
    return 0 


def get_x(feature, data, feature_type=None):
    """
    Utility function to help generalize the 
    way we get a feature value from any of the 
    feature files / data structures

    feature: Name of the feature category
    feature_type: Name of the feature sub-category
    data: Data for a specific review's features
    """
    if feature == 'decision':
        return get_x_decision(data)
    elif feature == 'review_length':
        return get_x_review_length(data)
    elif feature == 'aspect_coverage':
        return get_x_aspect_coverage(data)
    elif feature == 'aspect_recall':
        return get_x_aspect_recall(data)
    elif feature == 'aspect_count':
        return get_x_aspect_count(data, feature_type)
    elif feature == 'aspect_found':
        return get_x_aspect_found(data, feature_type)
    elif feature == 'argument_count':
        return get_x_argument_count(data, feature_type)
    elif feature == 'argument_found':
        return get_x_argument_found(data, feature_type)
    else:
        return None


def add_feature_column(df, feature_data, feature, feature_type=None):
    """
    Add a column of values to the dataframe
    for any feature category or sub_category
    """
    val = []
    #################################################################
    # Name this feature column based on the 
    # feature category (argument, aspect, specificity)
    # and the feature sub-category (argument_request, etc.)
    #################################################################
    if feature_type:
        col = feature + '_' + str(feature_type)
    else:
        col = feature


    #################################################################
    # For each review id, append the feature value to a list
    # and set the dataframe column values as that list 
    #################################################################
    for review_id in df['review_id'].values:
        data = feature_data.get(review_id)
        if data is None:
            print("No data for ", review_id, "in feature", col)
        val.append(get_x(feature, data, feature_type))
    df[col] = val


def add_politeness_column(df, corpus, liwc_df):
    politeness_features = {}
    for review_id in df['review_id'].values:
        word_count = liwc_df.loc[review_id, 'WC']
        if 'NIPS' in review_id:
            review_id = 'Review_' + review_id
        try:
            utt = corpus.get_object('utterance', review_id)
            # Binary politeness features
            # politeness_map = utt.meta['politeness_strategies']
            # for k, v in politeness_map.items():
            #     if k not in politeness_features:
            #         politeness_features[k] = []
            #     politeness_features[k].append(v)
            
            # Normalized count based politeness features
            # Format: [token, sentence index, sentence position]
            politeness_map = utt.meta['politeness_markers']
            for k, v in politeness_map.items():
                k = k[21:-2]
                if k not in politeness_features:
                    politeness_features[k] = []
                try:
                    value = len(v) / word_count
                except ZeroDivisionError:
                    value = 0
                politeness_features[k].append(value)
        except:
            # print("Politeness error: ", review_id)
            for k in politeness_features:
                politeness_features[k].append(0)
    
    for col in politeness_features:
        df[col] = politeness_features[col]


def get_labeled_features_df():
    # Load gold annotations dataframe
    file_path = '../data/gold_annotations.csv'
    df = pd.read_csv(file_path)
    df['venue'] = df['review_id'].apply(
        lambda x: 'neurips18' if 'NIPS' in x else 'iclr18'
    )
    # TODO: get venues from unique venues in df
    venues = ['iclr18', 'neurips18']
    return get_features(df, venues)


def get_unlabeled_features_df(venue):
    file_path = '../data/unlabeled/{venue}.csv'.format(venue=venue)
    df = pd.read_csv(file_path, index_col=0)
    df = df.reset_index().rename({'index': 'review_id'}, axis=1)
    df['venue'] = venue.split('_')[0]
    return get_features(df, [venue.split('_')[0]])


def get_features(df, venues):
    """
    This is where the features are actually getting added

    If you want to skip adding a particular feature altogether
    change the code / comment out the specific line here
    """
    #Load features data
    argument_data = get_argument_data(venues)
    aspect_data = get_aspect_data(venues)
    specificity_data = get_all_specificity_data(venues)
    ps_corpus = get_politeness_data(venues)
    liwc_df = get_liwc_data(venues)

    # Add feature columns to it
    aspect_feature_types = ['aspect_coverage']#, 'aspect_recall']
    aspect_agg_types = ['aspect_count']#, 'aspect_found']
    argument_agg_types = ['argument_count']#, 'argument_found']

    aspect_types = [
        'clarity', 'meaningful_comparison', 'motivation', 'originality',
        'replicability', 'soundness', 'substance', 'summary'
    ]
    argument_types = ['evaluation', 'fact', 'quote', 'reference', 'request']

    # Add general features
    # add_feature_column(df, argument_data, 'review_length')
    df['review_length'] = df['review_id'].apply(
        lambda x: len(specificity_data.get(x))
    )
    df['review_length'] = np.log(df['review_length'])
    
    # Add word count from liwc
    df['word_count'] = df['review_id'].apply(
        lambda x: liwc_df.loc[x, 'WC']
    )
    df['word_count'] = np.log(df['word_count'])

    # Add specificity features
    df['mean_specificity'] = df['review_id'].apply(
        lambda x: np.mean(specificity_data.get(x))
    )
    df['median_specificity'] = df['review_id'].apply(
        lambda x: np.median(specificity_data.get(x))
    )
    df['max_specificity'] = df['review_id'].apply(
        lambda x: min(1, (max(specificity_data.get(x))))
    )
    df['min_specificity'] = df['review_id'].apply(
        lambda x: min(specificity_data.get(x))
    )
    df['ratio_specificity'] = df['review_id'].apply(
        lambda x: get_specificity_ratio(specificity_data.get(x))
    )
    
    # Add aspect features
    for feature in aspect_feature_types:
        add_feature_column(df, aspect_data, feature)
    for aspect in aspect_types:
        for feature in aspect_agg_types:
            add_feature_column(df, aspect_data, feature, aspect)
        
    # Add argument features
    for argument in argument_types:
        for feature in argument_agg_types:
            add_feature_column(df, argument_data, feature, argument) 

    # Add politeness features
    add_politeness_column(df, ps_corpus, liwc_df)

    # Add summary liwc features
    for col in liwc_df.columns[2:6]:
        df['liwc_' + col] = df['review_id'].apply(
            lambda x: liwc_df.loc[x, col]
        )

    # Add token count based liwc features
    for col in liwc_df.columns[7:]:
        df['liwc_' + col] = df['review_id'].apply(
            lambda x: liwc_df.loc[x, col] / liwc_df.loc[x, 'WC']
        )   
    
    return df


def main():
    import optparse

    # process command-line arguments
    parser = optparse.OptionParser()
    parser.add_option(
        '-l', dest='labeled', action='store_true',
        default=True,
        help="Load labeled dataset (True / False) ?",
    )
    parser.add_option(
        '-u', dest='labeled', action='store_false',
        help="Load labeled dataset (True / False) ?",
    )
    
    parser.add_option(
        "-v",
        "--venue",
        dest="venue",
        default='iclr18',
        type="string",
        help="Venue for unlabeled dataset",
    )
    (options, args) = parser.parse_args()
    # print(options.labeled)
    if bool(options.labeled):
        df = get_labeled_features_df()
        df.to_csv('../data/features/common_ann.csv', index=False)
    else:
        df = get_unlabeled_features_df(options.venue)
        df.to_csv(
            '../data/features/unlabeled_{venue}.csv'.format(
                venue=options.venue
            ),
            index=False,
        )


if __name__ == "__main__":
    main()
