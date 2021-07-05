import torch
import json
import re
import nltk
import glob
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

data_dir = '../data/json_data/'
output_dir = '../data/arguments/'

filepaths = {
    'ICLR_2018': data_dir + 'iclr18.json',
    'ICLR_2019': data_dir + 'iclr19.json',
    'NeurIPS_2018': data_dir + 'neurips18.json',
    'NeurIPS_2019': data_dir + 'neurips19.json',
}

# data_df = pd.DataFrame(columns=['id', 'number', 'review_sent', 'venue', 'decision'])

def get_sentences(review):
    text = re.sub(r'\n', ' ', review)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\d.', '<NUM>', text)
    text = re.sub(r'al.', 'al', text)
    review_sentences = nltk.sent_tokenize(text)
    return review_sentences


def get_unlabeled_data(filepath=None):
    """
    Converts json file(s) containing entire reviews
    into a single dataframe contaning review sentences

    This is the input data for the model
    """
    data_dict = {}
    count = 0
    if filepath is not None:
        # process a single file
        process_review_file(filepath, data_dict, count, venue=None)
    else:
        # Process all reviews from a list of all conferences
        # stored at the top of this file
        for venue in tqdm(filepaths):
            path = filepaths[venue]
            count += process_review_file(filepath, data_dict, count, venue)
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')

    # if processing all conferences, save unlabeled input data
    if filepath is not None:
        data_df.to_csv(output_dir + 'unlabeled.csv')
    
    # Either ways, return the dataframe
    return data_df


def process_review_file(filepath, data_dict, count, venue):
    """
    Iteratively split a review into sentences, 
    add the sentences to a shared dictionary,
    and return the updated count 
    
    filepath: path to the JSON file of reviews
    data_dict: dictionary that stores all the sentences
    count: total number of sentences in the dictionary
    venue: conference venue (where the reviews are from)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        reviews = data['review_rebuttal_pairs']
        for number, review in tqdm(enumerate(reviews)):
            review_sentences = get_sentences(review['review_text']['text'])
            assert len(review_sentences) > 0
            for sent in review_sentences:
                data_dict[count] = {
                    'id': review['review_sid'],
                    'number': review['index'],
                    'review_sent': sent,
                    'venue': venue,
                    'decision': review['decision']
                }
                count += 1
    return count


def run_arguments_model(model_type, data_df, save_all=False):
    """
    Runs the trained argument models in evaluation mode
    """
    model_dict = {
        'bert': {
            'base': "bert-base-uncased",
            'saved': '../models/argument/Bert.model',
        },
        'scibert': {
            'base': "allenai/scibert_scivocab_uncased",
            'saved': '../models/argument/SciBert.model',
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(
        model_dict[model_type]['base'],
        num_labels=6,
        output_attentions=False,
        output_hidden_states=False)

    model.to(device)
    model.load_state_dict(torch.load(
        model_dict[model_type]['saved'], 
        map_location=torch.device('cpu')
    ))

    tokenizer = BertTokenizer.from_pretrained(
        model_dict[model_type]['base'],
        do_lower_case=True
    )
    encoded_data_test = tokenizer.batch_encode_plus(
        data_df.review_sent.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256,
        truncation=True,
        return_tensors='pt'
    )

    batch_size = 16

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    dataset_test = TensorDataset(input_ids_test, attention_masks_test)
    dataloader_test = DataLoader(
        dataset_test,
        sampler=SequentialSampler(dataset_test),
        batch_size=batch_size
    )

    model.eval()
    predictions= []
    for batch in tqdm(dataloader_test):    
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
        }
        with torch.no_grad():        
            output = model(**inputs)
        logits = output[0]
        predictions.append(logits)

    predictions = torch.cat(predictions, dim=0)
    probs = F.softmax(predictions, dim=1).cpu().numpy()

    label_dict_revert = {
        0: 'fact',
        1: 'evaluation',
        2: 'request',
        3: 'reference',
        4: 'non-arg',
        5: 'quote',
    }
    labels = []
    for p in probs:
        labels.append(label_dict_revert[np.argmax(p)])
    data_df['labels'] = labels
    if save_all:
        data_df.to_csv(output_dir + 'predictions.csv')
    else:
        data_df.to_csv(output_dir + 'predictions_test.csv')


def main():
    import optparse

    # process command-line arguments
    parser = optparse.OptionParser()
    parser.add_option(
        "-f",
        "--file",
        dest="filepath",
        default=None,
        type="string",
        help="Path to a specific json file with review data",
    )
    parser.add_option(
        "-m",
        "--model",
        dest="model_type",
        default='scibert',
        type="string",
        help="Type of model to be used for prediction (scibert/bert)?",
    )
    (options, args) = parser.parse_args()

    # Either process a single file
    # or process all venues in the dictionary 
    # called "filepaths" at the top of this file
    if options.filepath is not None:
        # TODO: check if path exists
        df = get_unlabeled_data(filepath=options.filepath)
        run_arguments_model(options.model_type, df, save_all=False)
    else:
        df = get_unlabeled_data(filepath=None)
        run_arguments_model(options.model_type, df, save_all=True)


if __name__ == "__main__":
    main()
