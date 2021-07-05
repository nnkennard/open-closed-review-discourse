import json
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

from nltk import agreement
from nltk.metrics import interval_distance, binary_distance

data_paths = {
    'iclr18': '../data/json_data/iclr18_ann.json',
    'neurips18': '../data/json_data/neurips18_ann_sm.json'
}
binary_cols = [
    'importance', 'originality', 'method', 
    'presentation', 'interpretation', 'reproducibility'
]
categorical_cols = binary_cols + ['metareview']

annotation_types = [
    'constructiveness', 'evidence', 'importance',
    'interpretation', 'method', 'originality',
    'presentation', 'reproducibility', 'overall', 'metareview'
]
likert_cols = ['overall', 'evidence', 'constructiveness']
all_anns = ['AS', 'DP', 'CL', 'SB']


def get_sorted_review_ids(venue):
    assert venue in data_paths

    review_ids = []
    with open(data_paths[venue], 'r') as f:
        data = json.load(f)
        for review in data['review_rebuttal_pairs']:
            review_ids.append(review['review_sid'])
    return sorted(review_ids)


def get_annotator_ids():
    iclr_ids = get_sorted_review_ids(venue='iclr18')
    neurips_ids = get_sorted_review_ids(venue='neurips18')
    AS_ids = iclr_ids[:149]
    DP_ids = iclr_ids[201:]
    SB_ids = neurips_ids[29:90]
    CL_ids = neurips_ids[90:]
    common_ids = iclr_ids[149:200] + neurips_ids[0:28]

    return {
        'AS': AS_ids,
        'DP': DP_ids,
        'SB': SB_ids,
        'CL': CL_ids,
        'common': common_ids
    }


################################################################################
# PART 1: Load and correct annotation json file
################################################################################
def check_binary(k, v):
    # Convert binary column values to range 0-1
    try:
        val = int(v)
        if k.lower() in binary_cols:
            return int(val > 2)
        return val
    except:
        return v


def get_y_aspect(data, aspect_type, review_id):
    # Get the value of the annotation 
    # without getting confused by missing keys
    try:
        return int(data['ratings'][aspect_type])
    except ValueError:
        # Maybe the value cannot be casted to an integer?
        # (applicable for metareview anntations)
        return data['ratings'][aspect_type]
    except:
        try:
            # Maybe the key is not all lower case?
            aspect_type = aspect_type[0].upper() + ''.join(aspect_type[1:])
            return int(data['ratings'][aspect_type])
        except:
            # Maybe the key is missing, because of a missing annotation?
            print("Missing key: ", aspect_type, "for ID", review_id)
            return None


def load_review_id_to_index_map(venues):
    review_data_dict = {}
    for venue in venues:
        assert venue in data_paths
        # Load review metadata
        with open(data_paths[venue], 'r') as file:
            review_data = json.load(file)
        for data in review_data['review_rebuttal_pairs']:
            review_data_dict[data['review_sid']] = int(data['index'])
    return review_data_dict


def load_annotations(file_path):
    # Load annotation results
    with open(file_path, 'r') as file:
        annotation_data = json.load(file)
    for data in annotation_data:
        # Load json ratings
        data['fields']['ratings'] = json.loads(data['fields']['ratings'])
    return annotation_data
    
METAREVIEW = "metareview"
BINARY_FIELDS = ("importance originality method presentation interpretation "
                  "reproducibility").split()

def metareview_cleanup(key, value):
    if key == value:
        return key
    elif key.lower() == METAREVIEW:
        return value
    else:
        assert value.lower() == METAREVIEW
        return key

def clean_ratings(ratings_json, pk):
    new_ratings = {}
    for k, v in ratings_json.items():
        try:
            int_value = int(v)
            if k in BINARY_FIELDS:
                if pk < 28:
                    int_value = int(int_value > 2)
                else:
                    assert int_value in range(2)
            new_ratings[k.lower()] = int_value
        except ValueError: # Probably metareview field
            new_ratings[METAREVIEW] = metareview_cleanup(k, v)
    return new_ratings

def correct_annotations(annotation_data):
  cleaned_annotation_obj = []
  for i in annotation_data:
    i["fields"]["ratings"] = clean_ratings(i["fields"]["ratings"], i["pk"])
    cleaned_annotation_obj.append(i)
  return cleaned_annotation_obj


def get_latest_annotations_only(annotation_data):
    # if a particular annotator submits 2 responses
    # for the same review id, then consider only the
    # last annotation response in our final annotation
    # data.
    #
    # This will remove duplicate annotations by the same 
    # annotator for the same review id
    #
    latest_annotations = {}
    line_num = 0
    for data in annotation_data:
        line_num += 1
        # if line_num <= 28:
        #     continue
        # Only keep the latest annotations
        review_id = data['fields']['review_id']
        ann = data['fields']['annotator_initials']
        
        if review_id not in latest_annotations:
            latest_annotations[review_id] = {}
        latest_annotations[review_id][ann] = data['fields']
    return latest_annotations


def get_annotations(file_path):
    annotation_data = load_annotations(file_path)
    corrected_data = correct_annotations(annotation_data)
    return get_latest_annotations_only(corrected_data)


def get_annotation_dataframe(file_path):
    # get data
    annotations = get_annotations(file_path)
    # process and place into a dataframe
    features_dict = {}
    i = 0
    for review_id in annotations:
        for ann in annotations[review_id]:
            data = annotations[review_id][ann]
            features_dict[i] = {
                'review_id': review_id,
                'annotator_initials': ann,
            }
            
            for category in annotation_types:
                val = get_y_aspect(data, category, review_id)
                features_dict[i][category] = val
            i += 1

    df = pd.DataFrame.from_dict(features_dict, orient='index')
    return df


################################################################################
# PART 2: Resolve Disagreements
################################################################################
def convert_metareview_to_numerical(df):
    # This conversion from string to int
    # is done so that that pivot table part
    # in get_best_annotator_trio(df) does
    # not throw an error.
    #
    # Is there a better way of doing things?
    #
    metareview_map = {
        'nota': 0, 
        'maybe': 1, 
        'no': 2, 
        'yes-agree': 3, 
        'yes-disagree': 4,
    }
    df['metareview'] = df['metareview'].apply(lambda x: metareview_map.get(x))
    return df


def get_best_annotator_trio(df):
    common_ids = get_annotator_ids()['common']
    df = convert_metareview_to_numerical(df.copy())
    # Find the best annotator trio for each binary category
    # So that we can resolve disagreements with voting
    best_annotator_trio = {}
    for annotation_category in categorical_cols:
        best_agreement = -1
        for anns in combinations(all_anns, 3):
            res_df_cat_each = df[df['review_id'].isin(common_ids)].pivot_table(
                annotation_category, ['review_id'], 'annotator_initials'
            )
            values = res_df_cat_each.median(axis=0)
            r_df = res_df_cat_each.fillna(values)

            formatted_codes = []
            for i, row in enumerate(r_df.iterrows()):
                row = row[1]
                for j, ann in enumerate(anns):
                    formatted_codes.append(
                        (ann,i,int(row[ann]))
                    )
            task = agreement.AnnotationTask(
                data=formatted_codes,
                distance=interval_distance
            )
            alpha = task.alpha()
            if alpha > best_agreement:
                best_annotator_trio[annotation_category] = anns
                best_agreement = alpha
    return best_annotator_trio


# def check_pair_vote(anns, labels_dict):
#     if anns[0] in labels_dict and anns[1] in labels_dict:
#         if labels_dict[anns[0]] == labels_dict[anns[1]]:
#             return labels_dict[anns[0]]
#     return None


def get_trio_vote(anns, labels_dict):
    labels = []
    for ann in anns:
        if ann in labels_dict:
            labels.append(labels_dict[ann])
    return Counter(labels).most_common(1)[0][0]


def vote(labels_dict, category, best_annotator_trio):
    """
    Resolve disagreements in binary labels
    """
    labels = labels_dict.values()
    counts = Counter(labels)
    k = list(counts)
    # If there are no disagreements
    if len(k)==1:
        return k[0]
    # If disagreement exists in the binary label
    # and there is a tie
    if len(k)==2 and counts[k[0]] == counts[k[1]]:
        return get_trio_vote(best_annotator_trio[category], labels_dict)
    return counts.most_common(1)[0][0]


def resolve_disagreement(review_df, annotation_category, best_annotator_trio):
    # resolve binary disagreements by voting
    if annotation_category in categorical_cols:
        return vote(
            review_df[annotation_category].to_dict(),
            annotation_category,
            best_annotator_trio,
        )
    # resolve likert disagreements by taking median
    elif annotation_category in likert_cols:
        annotation_values = [
            v for v in review_df[annotation_category].values if not pd.isna(v)
        ]
        return np.median(annotation_values)


def save_gold_annotations(annotation_df, file_path):
    count = 0
    resolved_review_ids = set()
    best_annotator_trio = get_best_annotator_trio(annotation_df)
    gold_annotations = {}

    for row in annotation_df.iterrows():
        row = row[1]
        r_id = row[0]
        # Only process unique review ids once
        # even though the annotation df has 4 rows
        # for each review id - one for each annotator
        if r_id not in resolved_review_ids:
            resolved_review_ids.add(r_id)
        else:
            continue
        gold_annotations[r_id] = {
            'annotator_initials': 'gold',
        }
        # get all 4 (or less) annotations for this review id
        review_df = annotation_df[annotation_df['review_id'] == r_id]
        review_df.index = review_df['annotator_initials']     
        # resolve disagreements for all categories for this review_id
        for annotation_category in review_df.columns[2:]:
            gold_annotations[r_id][annotation_category] = resolve_disagreement(
                review_df, annotation_category, best_annotator_trio
            )
        count += 1

    # Save final results in the csv format
    gold_df = pd.DataFrame.from_dict(gold_annotations, orient='index')
    gold_df = gold_df.reset_index().rename(columns={"index": "review_id"})
    gold_df.to_csv(file_path, index=False)


def main():
    file_path = "../data/harbor_annotation_0512.json"
    df = get_annotation_dataframe(file_path)
    save_gold_annotations(df, '../data/gold_annotations.csv')


if __name__ == "__main__":
    main()
