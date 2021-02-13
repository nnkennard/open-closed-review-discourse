from collections import defaultdict
import openreview
from tqdm import tqdm
import nltk
nltk.download('punkt')

from convokit import Corpus, Speaker, Utterance


speakers_corpus = {}
utterance_corpus = []


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
  ANON = "Anonymous"
  NAMED = "Named"
  PC = "ProgramChair"


def get_note_type(invitation):
    if "Submission" in invitation:
        return CommentCategories.SUBMISSION
    if "Official_Review" in invitation:
        return CommentCategories.REVIEW
    elif "Decision" in invitation:
        return CommentCategories.DECISION
    elif "Comment" in invitation:
        return CommentCategories.COMMENT


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
  else:
    assert author.startswith("~")
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
        speaker = Speaker(None, id, name, meta)
        speakers_corpus[id] = speaker
    return speaker


def get_base_utterance(note):
    meta = {
        'type': get_note_type(note.invitation),
        'forum': note.forum,
        'tcdate': note.tcdate,
        'tmdate': note.tmdate,
        'invitation': note.invitation,
        'number': note.number,
        'original': note.original,
        'title': note.content.get('title')
    }
    return Utterance(id=note.id,
                     conversation_id=note.forum,
                     reply_to=note.replyto,
                     timestamp=note.tcdate,
                     meta=meta)

def get_utterance_from_submission(note):
    # this id format is chosen to link submission to other author coments
    # id = submission.signatures[0] + '/Paper' + submission.number + '/Authors'
    speaker = get_speaker(note)
    utterance = get_base_utterance(note)

    utterance.speaker = speaker
    utterance.text = note.content['abstract']
    
    for key in ['keywords', 'TL;DR', 'pdf', 'code',
                'paperhash', 'original_pdf', '_bibtex']:
        utterance.add_meta(key, note.content.get(key))

    return utterance


def get_utterance_from_comment(note):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note)

    utterance.speaker = speaker
    utterance.text = note.content.get('comment')

    return utterance


def get_utterance_from_review(note):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note)

    utterance.speaker = speaker
    utterance.text = note.content.get('review')
    for key in note.content:
        if key != "review":
            utterance.add_meta(key, note.content.get(key))
    return utterance


def get_utterance_from_decision(note):
    # create speaker
    speaker = get_speaker(note)
    utterance = get_base_utterance(note)

    utterance.speaker = speaker
    utterance.text = note.content.get('comment')
    utterance.add_meta('decision', note.content.get('decision'))

    return utterance

def download_iclr20(client, forum=None):
    '''
    Main function for loading ICLR 2020 data into convokit
    '''
    print('getting ICLR 2020 data...')

    # get all ICLR '20 submissions, reviews, and meta reviews, and organize them by forum ID
    # (a unique identifier for each paper; as in "discussion forum").
    submissions = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2020/Conference/-/Blind_Submission', forum=forum)
    submissions_by_forum = {n.forum: n for n in submissions}

    # There is typically 3 reviews per forum.
    print('getting ICLR 2020 official reviews...')
    reviews = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2020/Conference/Paper.*/-/Official_Review', forum=forum)
    reviews_by_forum = defaultdict(list)
    for review in tqdm(reviews):
        reviews_by_forum[review.forum].append(review)

    # There can be many comments per forum.
    print('getting ICLR 2020 public and offical comments...')
    comments = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2020/Conference/Paper.*/-/.*Comment', forum=forum)
    comments_by_forum = defaultdict(list)
    for comment in tqdm(comments):
        comments_by_forum[comment.forum].append(comment)

    # Because of the way the Program Chairs chose to run ICLR '20,
    # decisions are taken directly from Decision Node.
    decisions = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2020/Conference/Paper.*/-/Decision', forum=forum)
    decisions_by_forum = {n.forum: n for n in decisions}

    # Build utterances using each comment, review, decision and submission
    # on the various ICLR 2020 forums
    for forum in tqdm(submissions_by_forum):
        forum_reviews = reviews_by_forum[forum]
        for review in forum_reviews:
            utterance_corpus.append(get_utterance_from_review(review))

        forum_comments = comments_by_forum[forum]
        for comment in forum_comments:
            utterance_corpus.append(get_utterance_from_comment(comment))

        forum_decision = decisions_by_forum[forum]
        utterance_corpus.append(get_utterance_from_decision(forum_decision))

        submission = submissions_by_forum[forum]
        utterance_corpus.append(get_utterance_from_submission(submission))


def main():
    client = openreview.Client(baseurl='https://api.openreview.net')
    download_iclr20(client)
    corpus = Corpus(utterances=utterance_corpus)
    corpus.print_summary_stats()
    corpus.dump("iclr20_corpus", base_path='./data/convokit/')

if __name__ == "__main__":
    main()
