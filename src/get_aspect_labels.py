import json
import collections

def get_labels():
    # Get aspect labels
    venues = ['neurips18', 'iclr18']
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

        # Link aspects to review id
        for review_id in index_data:
            r_idx = index_data[review_id]['review']
            aspect_data[review_id] = {
                'review_aspects': aspect_labels[r_idx],
            }

    # Get review text
    review_text = {}
    with open('../data/json_data/neurips18.json', 'r') as f:
        data = json.load(f)

    for i in range(len(data['review_rebuttal_pairs'])):
        review_text[data['review_rebuttal_pairs'][i]['review_sid']] = data['review_rebuttal_pairs'][i]['review_text'][
            'text']

    with open('../data/json_data/iclr18.json', 'r') as f:
        data = json.load(f)

    for i in range(len(data['review_rebuttal_pairs'])):
        review_text[data['review_rebuttal_pairs'][i]['review_sid']] = data['review_rebuttal_pairs'][i]['review_text'][
            'text']

    for id in aspect_data:
        try:
            aspect_data[id]['text'] = review_text[id]
        except:
            continue

    return aspect_data

def get_aspect_spans(labels, review_text):
    spans = collections.defualtdict(list)
    for i in range(len(labels)):
        start = labels[i][0]
        end = labels[i][1]
        spans[labels[i][2]].append(review_text[start: end])

    return spans

def save_spans():
    aspect_labels = get_labels()
    for review_id in aspect_labels:
        aspect_labels[review_id]['spans'] = get_aspect_spans(aspect_labels[review_id]['review_aspects'],
                                                             aspect_labels[review_id]['text'])

    with open('../data/aspect_labels_and_spans.json', 'w') as f:
        json.dump(aspect_labels, f)

if __name__ == "__main__":
    save_spans()
