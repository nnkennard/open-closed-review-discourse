import json
import re


data_dir = '../data/json_data/'

filepaths = {
    # 'acl17': data_dir + 'acl17.json', 
    # 'conll16': data_dir + 'conll16.json',
    'iclr18': data_dir + 'iclr18.json',
    'iclr19': data_dir + 'iclr19.json',
    'neurips18': data_dir + 'neurips18.json',
    'neurips19': data_dir + 'neurips19.json',
}

review_data = {}
write_dir = '../data/aspect_tagger/'


def write_aspect_input(fp, text):
    # convert each review into 
    # a single line for the aspect input file
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r' +', ' ', text)
    fp.write(text + '\n')


def load_review_data(review_data, venue, filepath=None):
    # Load review data and metadata from the json file
    if filepath is not None:
        path = filepath
    else:
        path = filepaths[venue]
    with open(path, 'r') as file:
        review_data[venue] = json.load(file)


def convert_review_json_to_aspect_input(review_data, venue):
    """
    Convert review and meta-review text in the source JSON
    to a single line of text in the aspect input file
    """
    idx = -1
    aspect_idx = {}
    
    with open(write_dir + venue + "_aspect_input.txt", 'w+') as file:
        for data in review_data[venue]['review_rebuttal_pairs']:
            text = data['review_text']['text']
            write_aspect_input(file, text)

            # Record the line index in the aspect input file
            # for this review and meta-review text
            # and write it in a JSON file

            # Keep track of which review id
            # is on which line of the input file
            idx += 1
            aspect_idx[data['review_sid']] = {
                'review': idx,
                'meta_review': None,
            }
            
            # Also keep track of the meta-review line index
            # only if the meta-review is in the aspect input file
            text = data['meta_review'].get('text')
            if text:
                write_aspect_input(file, text)
                idx += 1
                aspect_idx[data['review_sid']]['meta_review'] = idx

    with open(write_dir + venue + "_aspect_idx.json", 'w+') as f:
        json.dump(aspect_idx, f)



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
        "-v",
        "--venue",
        dest="venue",
        default='test',
        type="string",
        help="Conference venue (where the review data is from like 'iclr18')",
    )
    (options, args) = parser.parse_args()

    # Either process a single file
    # or process all venues in the dictionary 
    # called "filepaths" at the top of this file
    if options.filepath is not None:
        # TODO: check if path exists
        load_review_data(review_data, options.venue, options.filepath)
        convert_review_json_to_aspect_input(review_data, options.venue)
    else:
        for venue in tqdm(filepaths):
            load_review_data(review_data, venue, filepath=None)
            convert_review_json_to_aspect_input(review_data, venue)


if __name__ == "__main__":
    main()
