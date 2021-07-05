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
speakers_corpus = {}

# 2018 - 2021 get forums like this
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

class CommentCategories(object):
  SUBMISSION = "Submission"
  REVIEW = "Review"
  DECISION = "Decision"
  COMMENT = "Comment"
  WITHDRAW = "Withdraw"
  ETHICS_REVIEW = "Ethics Review"


class AuthorCategories(object):
  CONFERENCE = "Conference"
  AUTHOR = "Author"
  AC = "AreaChair"
  REVIEWER = "Reviewer"
  ANON = "Anonymous"
  NAMED = "Named"
  PC = "ProgramChair"
  ETHICS = "EthicsCommittee"


def get_note_type(invitation):
    if "/Blind_Submission" in invitation:
        return CommentCategories.SUBMISSION
    if "/Official_Review" in invitation:
        return CommentCategories.REVIEW
    elif "/Meta_Review"  in invitation or "/Decision" in invitation:
        return CommentCategories.DECISION
    elif "Comment" in invitation:
        return CommentCategories.COMMENT
    elif "Withdraw" in invitation:
        return CommentCategories.WITHDRAW
    elif "Ethics_Meta_Review" in invitation:
        return CommentCategories.ETHICS_REVIEW


def get_author_type(author):
  assert "|" not in author # Only one signature per comment, I hope
  if "Author" in author:
    return AuthorCategories.AUTHOR
  if author.endswith("/Conference"):
    return AuthorCategories.CONFERENCE
  elif "Area_Chair" in author:
    return AuthorCategories.AC
  elif "Program_Chairs" in author:
    return AuthorCategories.PC
  elif "AnonReviewer" in author:
    return AuthorCategories.REVIEWER
  elif author == "(anonymous)":
    return AuthorCategories.ANON
  elif "Ethics_Committee" in author:
      return AuthorCategories.ETHICS
  else:
    if not author.startswith("~"):
        print("No Author Type Found:", author)
    # assert author.startswith("~")
    return AuthorCategories.NAMED


def get_iclr_submissions(client, year):
    # get all ICLR '20 submissions, reviews, and meta reviews, and organize them by forum ID
    # (a unique identifier for each paper; as in "discussion forum").
    submissions = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2020/Conference/-/Blind_Submission')
    submissions_by_forum = {n.forum: n for n in submissions}


def get_speaker(note):
    id = note.signatures[0]
    if id in speakers_corpus:
        speaker = speakers_corpus[id]
    else:
        speaker_type = get_author_type(note.signatures[0])
        meta = {
            'forum': note.forum,
            'type': speaker_type,
        }
        if speaker_type == AuthorCategories.CONFERENCE:
            meta['authors'] = note.content['authors']
            meta['authorids'] = note.content['authorids']
            
        name = note.signatures[0]
        speaker = Speaker(owner=None, id=id, name=name, meta=meta)
        speakers_corpus[id] = speaker
    
    assert speaker is not None
    assert speaker.meta is not None

    return speaker


def get_base_utterance(note, speaker, decision):
    meta = {
        'type': get_note_type(note.invitation),
        'forum': note.forum,
        'tcdate': note.tcdate,
        'tmdate': note.tmdate,
        'invitation': note.invitation,
        'number': note.number,
        'original': note.original,
        'title': note.content.get('title'),
        'decision': decision,
    }
    return Utterance(id=note.id, 
                     speaker=speaker,
                     conversation_id=note.forum,
                     reply_to=note.replyto,
                     timestamp=note.tcdate,
                     meta=meta)

def get_utterance_from_submission(note, decision):
    # this id format is chosen to link submission to other author coments
    # id = submission.signatures[0] + '/Paper' + submission.number + '/Authors'
    speaker = get_speaker(note)
    utterance = get_base_utterance(note, speaker, decision)
    utterance.text = note.content['abstract']
    
    for key in ['keywords', 'TL;DR', 'pdf', 'code',
                'paperhash', 'original_pdf', '_bibtex']:
        utterance.add_meta(key, note.content.get(key))

    return utterance


def get_utterance_from_comment(note, decision):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note, speaker, decision)
    utterance.text = note.content.get('comment')

    return utterance


def get_utterance_from_review(note, decision):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note, speaker, decision)
    utterance.text = note.content.get('review')
    for key in note.content:
        if key != "review":
            utterance.add_meta(key, note.content.get(key))
    return utterance


def get_utterance_from_withdraw(note, decision):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note, speaker, decision)
    utterance.text = note.content.get('withdrawal confirmation')

    return utterance


def get_utterance_from_decision(note):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note, speaker, note.content.get('decision'))
    utterance.text = note.content.get('comment')

    return utterance


def get_utterance_from_ethics_review(note):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note, speaker, note.content.get('decision'))
    utterance.text = note.content.get('ethics_review')

    return utterance


def get_utterance_from_forum(note, decision):
    note_type = get_note_type(note.invitation)

    if note_type == CommentCategories.REVIEW:
        return get_utterance_from_review(note, decision)
    elif note_type == CommentCategories.SUBMISSION:
        return get_utterance_from_submission(note, decision)
    elif note_type == CommentCategories.COMMENT:
        return get_utterance_from_comment(note, decision)
    elif note_type == CommentCategories.DECISION:
        return get_utterance_from_decision(note)
    elif note_type == CommentCategories.ETHICS_REVIEW:
        return get_utterance_from_ethics_review(note)
    elif note_type == CommentCategories.WITHDRAW:
        return get_utterance_from_withdraw(note, decision)
    else:
        print(note.invitation)
        return None

def download_iclr(client, year, limit=None, forum=None):
    '''
    Main function for loading ICLR data into convokit
    '''
    utterance_list = []

    print('getting ICLR data to make a corpus...')
    # get all ICLR submissions, reviews, and meta reviews, and organize them by forum ID
    # (a unique identifier for each paper; as in "discussion forum").
    submissions = openreview.tools.iterget_notes(
        client, invitation=submission_invitations[year], forum=forum)
    submissions_by_forum = {n.forum: n for n in submissions}

    # Decisions are taken directly from Decision Node.
    decisions = openreview.tools.iterget_notes(
        client, invitation=decision_invitations[year], forum=forum)
    decisions_by_forum = {n.forum: n for n in decisions}

    ignored_forums = []

    # Build utterances using each comment, review, decision and submission
    # on the various ICLR 2020 forums
    for idx, forum in tqdm(enumerate(submissions_by_forum)):
        
        # Get only few forums to test
        if limit is not None and idx >= limit:
            break

        # We consider each forum as a single conversation
        # conversation = Conversation()

        forum_decision = decisions_by_forum[forum]
        decision = forum_decision.content.get('decision')

        utterance_ids = set()
        reply_ids = set()

        forum_notes = client.get_notes(
            invitation='ICLR.cc/{year}/Conference/.*'.format(year=year), forum=forum)
        
        # Check if replyto is valid
        # if replyto chain has missing comment, then conversation tree 
        # cannot be built
        # ignore all forums with such missing comments
        for note in tqdm(forum_notes, disable=True):
            utterance_ids.add(note.id)
            if note.replyto is None:
                continue
            if note.replyto not in utterance_ids:
                reply_ids.add(note.replyto)
        if reply_ids - utterance_ids:
            ignored_forums.append(forum)
            continue
        
        # If all replyto comments are accounted for
        # Add utterances to the corpus
        for note in tqdm(forum_notes, disable=True):
            utterance = get_utterance_from_forum(note, decision)
            if utterance:
                # TODO: Add year, conference name to utterance?
                utterance_list.append(utterance)
    return utterance_list, ignored_forums



def main():
    import optparse

    # process command-line arguments
    parser = optparse.OptionParser()
    parser.add_option(
        '-y', '--year', dest='year',
        default=2018, type='int',
        help="Year to extract ICLR data from (2013-2021)?",
    )
    parser.add_option(
        '-l', '--limit', dest='limit',
        default=None, type='int',
        help="Limit number of forums extracted to this number (integer)",
    )
    (options, args) = parser.parse_args()

    utterance_list, ignored_forums = download_iclr(client, options.year, options.limit)
    corpus = Corpus(utterances=utterance_list)
    corpus.print_summary_stats()
    file_name = 'iclr' + str(options.year)[-2:] + '_corpus'
    corpus.dump(file_name, base_path='../data/convokit/')



if __name__ == "__main__":
    main()