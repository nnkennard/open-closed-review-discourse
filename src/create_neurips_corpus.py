import json
from tqdm import tqdm
from convokit import Corpus, Speaker, Utterance

speakers_corpus = {}

json_file_paths = {
    # 2013: None,
    # 2014: None,
    # 2015: None,
    2016: '../data/json_data/neurips16.json',
    2017: '../data/json_data/neurips17.json',
    2018: '../data/json_data/neurips18.json',
    2019: '../data/json_data/neurips19.json',
    # 2020: None,
}


class CommentCategories(object):
  SUBMISSION = "Submission"
  REVIEW = "Review"
  DECISION = "Decision"
  COMMENT = "Comment"


class AuthorCategories(object):
  CONFERENCE = "Conference"
  AUTHOR = "Author"
  AC = "AreaChair"
  REVIEWER = "Reviewer"


def get_utterance_type(text_type):
    if text_type == 'abstract':
        return CommentCategories.SUBMISSION
    if text_type == 'reviews':
        return CommentCategories.REVIEW
    elif text_type == 'meta_review':
        return CommentCategories.DECISION
    elif text_type == 'rebuttal':
        return CommentCategories.COMMENT


def get_author_type(utt_type):
  if utt_type == CommentCategories.COMMENT:
    return AuthorCategories.AUTHOR
  if utt_type == CommentCategories.SUBMISSION:
    return AuthorCategories.CONFERENCE
  elif utt_type == CommentCategories.DECISION:
    return AuthorCategories.AC
  elif utt_type == CommentCategories.REVIEW:
    return AuthorCategories.REVIEWER
  else:
    print("No Author Type Found:", author)
    return None


def get_id(obj_type, forum_id, number=None):
    if number is None:
        number = 0
    return "_".join([obj_type, forum_id, str(number)])


def get_reply_to(utterance_type, forum_id):
    if utterance_type == CommentCategories.SUBMISSION:
        # root of the conversation is the submitted paper
        return None
    else:
        # All text is a reply to the main submission
        return get_id(CommentCategories.SUBMISSION, forum_id)


def get_neurips_data(file_path):
    # get all NeurIPS abstracts, reviews, and meta reviews, and organize them
    # by forum ID (a unique identifier for each paper; as in "discussion forum")
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    submissions_by_forum = {}
    for review in data['review_rebuttal_pairs']:
        review_id = review['review_sid']
        assert 'NIPS' in review_id

        forum_id = '_'.join(review_id.split('_')[:-1])
        if forum_id not in submissions_by_forum:
            submissions_by_forum[forum_id] = {
                'abstract': review.get('abstract'),
                'reviews': [],
                'rebuttal': str(review['rebuttal_text']['text']),
                'meta_review': str(review['meta_review']['text']),
            }
        submissions_by_forum[forum_id]['reviews'].append(
            review['review_text']['text']
        )
    return submissions_by_forum


def get_speaker(utt_type, forum_id, speaker_id):
    speaker_type = get_author_type(utt_type)
    if speaker_id in speakers_corpus:
        speaker = speakers_corpus[speaker_id]
    else:
        meta = {
            'forum': forum_id,
            'type': speaker_type,
        }
        speaker = Speaker(owner=None, id=speaker_id, name=speaker_id, meta=meta)
        speakers_corpus[speaker_id] = speaker
    
    assert speaker is not None
    assert speaker.meta is not None

    return speaker


def get_utterance(text_type, forum_id, text, number=None):
    utterance_type = get_utterance_type(text_type)
    utterance_id = get_id(utterance_type, forum_id, number)
    speaker = get_speaker(utterance_type, forum_id, utterance_id)
    reply_to = get_reply_to(utterance_type, forum_id)

    meta = {
        'type': utterance_type,
        'forum': forum_id,
        'decision': 'Accept',
    }
    return Utterance(id=utterance_id, 
                     speaker=speaker,
                     conversation_id=forum_id,
                     reply_to=reply_to,
                     text=text,
                     timestamp=None,
                     meta=meta)



def download_neurips(year):
    '''
    Main function for loading NeurIPS data into convokit
    '''
    utterance_list = []

    print('getting NeurIPS data...')
    assert year in json_file_paths
    submissions_by_forum = get_neurips_data(json_file_paths[year])

    # Build utterances using each rebuttal, review, meta-review and abstract
    # on the various NeurIPS forums
    for forum_id in tqdm(submissions_by_forum):
        forum_notes = submissions_by_forum[forum_id]
        for key in tqdm(forum_notes, disable=True):
            note = forum_notes[key]
            if key == 'reviews':
                for i, review in enumerate(note):
                    number = i + 1
                    utterance_list.append(get_utterance(key, forum_id, review, number))
            else:
                utterance_list.append(get_utterance(key, forum_id, note))
    return utterance_list

def main():
    for year in tqdm(json_file_paths):
        utterance_list = download_neurips(year)
        corpus = Corpus(utterances=utterance_list)
        corpus.print_summary_stats()
        file_name = json_file_paths[year].split('/')[-1][:-5] + '_corpus'
        corpus.dump(file_name, base_path='../data/convokit/')

if __name__ == '__main__':
    main()