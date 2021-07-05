from collections import defaultdict

import json
from pprint import pprint
import re
import openreview
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from convokit import Corpus, Speaker, Utterance, Conversation

client = openreview.Client(baseurl='https://api.openreview.net')


submission_invitations = {
    2013: 'ICLR.cc/2013/conference/-/submission',
    2014: 'ICLR.cc/2014/conference/-/submission',
    2015: None,
    2016: None,
    2017: 'ICLR.cc/2017/conference/-/submission',
    2018: 'ICLR.cc/2018/Conference/-/Blind_Submission',
    2019: 'ICLR.cc/2019/Conference/-/Blind_Submission',
    2020: 'ICLR.cc/2020/Conference/-/Blind_Submission',
    2021: 'ICLR.cc/2021/Conference/-/Blind_Submission',
}

withdrawn_invitations = {
    2013: None,
    2014: None,
    2015: None,
    2016: None,
    2017: None,
    2018: 'ICLR.cc/2018/Conference/-/Withdrawn_Submission',
    2019: 'ICLR.cc/2019/Conference/-/Withdrawn_Submission',
    2020: 'ICLR.cc/2020/Conference/-/Withdrawn_Submission',
    2021: 'ICLR.cc/2021/Conference/-/Withdrawn_Submission',
}

review_invitations = {
    2013: None,
    2014: 'ICLR.cc/2014/-/submission/conference/review',
    2015: None,
    2016: None,
    2017: 'ICLR.cc/2017/conference/-/paper.*/official/review',
    2018: 'ICLR.cc/2018/Conference/-/Paper.*/Official_Review',
    2019: 'ICLR.cc/2019/Conference/-/Paper.*/Official_Review',
    2020: 'ICLR.cc/2020/Conference/Paper.*/-/Official_Review',
    2021: 'ICLR.cc/2021/Conference/Paper.*/-/Official_Review'
}

rebuttal_invitations = {
    2013: None,
    2014: None,
    2015: None,
    2016: None,
    2017: 'ICLR.cc/2017/conference/-/paper.*/official/comment',
    2018: 'ICLR.cc/2018/Conference/-/Paper.*/Official_Comment',
    2019: 'ICLR.cc/2019/Conference/-/Paper.*/Official_Comment',
    2020: 'ICLR.cc/2020/Conference/Paper.*/-/Official_Comment',
    2021: 'ICLR.cc/2021/Conference/Paper.*/-/Official_Comment'
}

comment_invitations = {
    2013: None,
    2014: None,
    2015: None,
    2016: None,
    2017: 'ICLR.cc/2017/conference/-/paper.*/public/comment',
    2018: 'ICLR.cc/2018/Conference/-/Paper.*/Public_Comment',
    2019: 'ICLR.cc/2019/Conference/-/Paper.*/Public_Comment',
    2020: 'ICLR.cc/2020/Conference/Paper.*/-/Public_Comment',
    2021: 'ICLR.cc/2021/Conference/Paper.*/-/Official_Comment'
}

decision_invitations = {
    2013: 'ICLR.cc/2013/conference/-/submission',
    2014: 'ICLR.cc/2014/conference/-/submission',
    2015: None,
    2016: None,
    2017: 'ICLR.cc/2017/conference/-/paper.*/acceptance',
    2018: 'ICLR.cc/2018/Conference/-/Acceptance_Decision',
    2019: 'ICLR.cc/2019/Conference/-/Paper.*/Meta_Review',
    2020: 'ICLR.cc/2020/Conference/Paper.*/-/Decision',
    2021: 'ICLR.cc/2021/Conference/Paper.*/-/Decision'
}

decision_keys = {
    2013: 'decision',
    2014: 'decision',
    2015: None,
    2016: None,
    2017: 'decision',
    2018: 'decision',
    2019: 'recommendation',
    2020: 'decision',
    2021: 'decision'
}

meta_review_keys = {
    2013: 'comment',
    2014: 'comment',
    2015: None,
    2016: None,
    2017: 'comment',
    2018: 'comment',
    2019: 'metareview',
    2020: 'comment',
    2021: 'comment'
}


def download_iclr(client, year, limit=None, forum=None):
    '''
    Main function for loading ICLR data into convokit

    If a specific forum id is specified, it will extract data from only that forum
    '''
    utterance_list = []
    if forum is not None:
        conf = 'iclr' + str(year)[-2:] + '_' + str(forum)
    else:
        conf = 'iclr' + str(year)[-2:]
        
    print('getting ICLR data...')
    # get all ICLR submissions, reviews, and meta reviews, and organize them by forum ID
    # (a unique identifier for each paper; as in "discussion forum").
    if submission_invitations.get(year):
        submissions = openreview.tools.iterget_notes(
            client, invitation=submission_invitations[year], forum=forum)
        submissions_by_forum = {n.forum: n for n in submissions}
    else:
        raise Exception(
            "Out of range: No submission invitation found for year " + year
        )

    # Decisions are taken directly from Decision Node.
    if decision_invitations.get(year):
        decisions = openreview.tools.iterget_notes(
            client, invitation=decision_invitations[year], forum=forum)
        decisions_by_forum = {n.forum: n for n in decisions}
    else:
        raise Exception(
            "Out of range: No decision invitation found for year " + year
        )

    forum_data = {}
    without_rebuttal_review_ids = set()

    data_dict = {
        'conference': conf,
        'split': 'traindev',
        'subsplit': 'train',
        'review_rebuttal_pairs': []
    }

    index = 0
    for idx, forum in tqdm(enumerate(submissions_by_forum)):
        
        # Get only a few forums if limit is set
        if limit is not None and idx >= limit:
            break

        decision = True
        submission_note = submissions_by_forum[forum]
        decision_note = decisions_by_forum.get(forum)
        
        # For any valid submission
        forum_data[forum] = {
            'title': submission_note.content.get('title'),
            'pdf': submission_note.content.get('pdf'),
            'content': submission_note.content,
        }

        # Unless submission was withdrawn, record its decision
        if decision_note:
            forum_data[forum]['decision'] = decision_note.content.get(
                decision_keys[year]
            )
            forum_data[forum]['meta_review'] = {
                'text': decision_note.content.get(meta_review_keys[year]),
                'title': decision_note.content.get('title'),
                'content': decision_note.content
            }
        else:
            # If decision is not found, check if this
            # submission was withdrawn
            decision = None
            if withdrawn_invitation.get(year):
                withdrawn_notes = client.get_notes(
                    forum=forum, 
                    invitation=withdrawn_invitations[year]
                )
                if len(withdrawn_notes) < 1:
                    # If submission was not withdrawn,
                    # and the decision is not found,
                    # flag this note
                    print("Not withdrawn: ", forum)
                else:
                    decision = 'Witndrawn'
            # Without the decision note, we cannot get meta-review
            forum_data[forum] = {
                'decision': decision,
                'meta_review': {
                    'text': None,
                    'title': None,
                }
            }

        # Get all reviews on this forum 
        try:
            if review_invitations.get(year):
                reviews = client.get_notes(
                    forum=forum,
                    invitation=review_invitations[year]
                )
            else:
                raise Exception(
                    "Out of range: No review invitation found for year " + year
                )
            for review in reviews:
                # Get all rebuttals that are a direct child of this review
                # or a direct child of this forum
                specific_rebuttals = []
                general_rebuttals = []
                specific_rebuttal_ids = []
                general_rebuttal_ids = []

                if rebuttal_invitations.get(year):
                    rebuttal_notes = client.get_notes(
                        forum=review.forum,
                        invitation=rebuttal_invitations[year]
                    )
                else:
                    rebuttal_notes= None

                for note in rebuttal_notes:
                    if 'Author' not in note.signatures[0]:
                        continue
                    if note.replyto == review.id:
                        specific_rebuttals.append(note)
                    elif note.replyto == forum:
                        general_rebuttals.append(note)

                specific_rebuttal_text = ''
                for rebuttal in specific_rebuttals:
                    specific_rebuttal_ids.append(rebuttal.id)
                    specific_rebuttal_text += rebuttal.content['comment']

                general_rebuttal_text = ''
                for rebuttal in general_rebuttals:
                    general_rebuttal_ids.append(rebuttal.id)
                    general_rebuttal_text += rebuttal.content['comment']
                
                rebuttal_ids = {
                    'specific': specific_rebuttal_ids,
                    'general': general_rebuttal_ids,
                }
                rebuttal_text = {
                    'specific': specific_rebuttal_text,
                    'general': general_rebuttal_text,
                }

                if len(specific_rebuttal_text) < 1 and len(general_rebuttal_text) < 1:
                    # If neither specific nor general rebuttal found
                    print("No rebuttal found: ", year, index, forum, review.id)
                    without_rebuttal_review_ids.add(review.id)
                    rebuttal_text = None

                # elif len(general_rebuttal_text) > 1:
                #     # If specific rebuttal not found but general rebuttal found
                #     print("General rebuttal found: ", year, index, forum, review.id)

                if not forum_data[review.forum]['meta_review']['text']:
                    print("No meta-review found: ", year, index, forum)
                    # continue
                
                # Get all revisions for this review note
                histories = []
                revision_notes = client.get_references(referent=review.id)
                for revision in revision_notes:
                    histories.append(
                        {
                            'note_id': revision.id,
                            'content': revision.content,
                        }
                    )

                review_dict = {
                    'index': index,
                    'review_sid': review.id,
                    'rebuttal_sid': rebuttal_ids,
                    'review_text': {
                        'sentences': [],
                        'text': review.content.get('review'),
                        'title': review.content.get('title')
                    },
                    'rebuttal_text': {
                        'sentences': [],
                        'text': rebuttal_text
                    },
                    'review_author': review.signatures[0].split('/')[-1],
                    'forum': review.forum,
                    'labels': {
                        'confidence': int(review.content.get('confidence')[0]),
                        'rating': int(review.content.get('rating')[0])
                    },
                    'exact_matches': [],
                    'histories': histories,
                }
                for key in forum_data[review.forum]:
                    review_dict[key] = forum_data[review.forum][key]

                data_dict['review_rebuttal_pairs'].append(review_dict)
                index += 1

        except Exception as e:
            print(index, id, str(e))
        # if index > 2: break
    with open('/content/drive/MyDrive/cs696/data/json_data/' + conf + '.json', 'w+') as f:
        json.dump(data_dict, f, default=str)
    
    return data_dict, without_rebuttal_review_ids


def main():
    import optparse

    # process command-line arguments
    parser = optparse.OptionParser()
    parser.add_option(
        '-y', '--year', dest='year',
        type='int', 
        default=2018,
        help="Year to extract ICLR data from (2013-2021)?",
    )
    parser.add_option(
        '-l', '--limit', dest='limit',
        default=None, type='int',
        help="Limit number of forums extracted to this number (integer)",
    )
    (options, args) = parser.parse_args()

    download_iclr(client, options.year, options.limit)


if __name__ == "__main__":
    main()