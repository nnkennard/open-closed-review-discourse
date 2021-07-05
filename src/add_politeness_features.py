from convokit import Corpus, Speaker, Utterance
from convokit.prompt_types import PromptTypeWrapper
from convokit import PolitenessStrategies
from convokit import TextParser

CORPUS_FILE_PATH = '../data/convokit/{venue}_corpus'
corpus_file_paths = {
    'iclr18': '../data/convokit/iclr18_corpus',
    'iclr19': '../data/convokit/iclr19_corpus',
    # 'iclr20': '../data/convokit/iclr20_corpus',
    'neurips18': '../data/convokit/neurips18_corpus',
    'neurips19': '../data/convokit/neurips19_corpus',
    # 'neurips20': '../data/convokit/neurips20_corpus',
}


def load_corpus(path):
    print(path)
    corpus = Corpus(path)
    corpus.print_summary_stats() 
    print('-'*50, '\n')
    return corpus


def get_politeness_features(corpus):
    parser = TextParser(verbosity=1000)
    corpus = parser.transform(corpus) 
    ps = PolitenessStrategies(verbose=1000)
    corpus = ps.transform(corpus, markers=True)
    return corpus


def main():
    import optparse

    # process command-line arguments
    parser = optparse.OptionParser()
    parser.add_option(
        "-c",
        "--conference",
        dest="conference",
        default='iclr',
        type="string",
        help="Conference for the dataset (e.g. iclr, neurips)",
    )
    parser.add_option(
        "-y",
        "--year",
        dest="year",
        default='2018',
        type="int",
        help="Year for the dataset (e.g. 2018, 2021)",
    )
    (options, args) = parser.parse_args()
    venue = options.conference.lower() + str(options.year)[-2:]
    path = CORPUS_FILE_PATH.format(venue=venue)
    corpus = load_corpus(path)
    corpus = get_politeness_features(corpus)
    new_path = venue + '_corpus_ps'
    corpus.dump(new_path, base_path='../data/convokit/')

if __name__ == '__main__':
    main()