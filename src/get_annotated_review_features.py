ann_df = pd.read_csv('../data/all_annotations.csv')

reviews = {}

with open('../data/json_data/iclr18_ann.json', 'r') as f:
    data = json.load(f)

for review in data['review_rebuttal_pairs']:
    r_id = review['review_sid']
    reviews[r_id] = review['review_text']['text']


with open('../data/json_data/neurips18_url.json', 'r') as f:
    data = json.load(f)

for review in data['review_rebuttal_pairs']:
    r_id = review['review_sid']
    reviews[r_id] = review['review_text']['text']

ann_df['review_text'] = ann_df['review_id'].apply(lambda x: reviews.get(x))

feat_df = gen_df.drop(columns=['annotator_initials', 'constructiveness', 'evidence',
       'importance', 'interpretation', 'method', 'originality', 'presentation',
       'reproducibility', 'overall', 'metareview', 'venue'], axis=1)

result_df = pd.merge(ann_df, feat_df, how='left', left_on='review_id', right_on='review_id').sort_values(by='review_id')
result_df.to_csv('../data/error_analysis/ann_with_feats.csv', index=False)