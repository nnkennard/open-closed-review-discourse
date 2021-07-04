import argparse
import collections
import json

parser = argparse.ArgumentParser(
    description='Clean and anonymize annotation data; adjudicate')
parser.add_argument('-a',
                    '--annotation_file',
                    type=str,
                    help='path to Django annotation dump')

BINARY_FIELDS = ('importance originality method presentation interpretation '
                 'reproducibility').split()
LIKERT_FIELDS = "overall evidence constructiveness".split()
ALL_FIELDS = BINARY_FIELDS + LIKERT_FIELDS

METAREVIEW = "metareview"
METAREVIEW_VALUES = "nota maybe no yes-agree yes-disagree".split()
METAREVIEW_MAP = {value: i for i, value in enumerate(METAREVIEW_VALUES)}

VALID_ANNOTATORS = "AS CL DP SB".split()
ANNOTATOR_MAP = {
    initials: "anno{0}".format(i)
    for i, initials in enumerate(sorted(VALID_ANNOTATORS))
}

GOLD_ANNOTATION = 'gold'

class Annotation(object):
  def __init__(self, annotations):
    self.valid_annotations = self.get_valid_annotations(annotations)
    print(annotator_set)

  def _get_valid_annotations(self, annotations)



def main():

  args = parser.parse_args()

  annotation_map = collections.defaultdict(list)

  with open(args.annotation_file, 'r') as f:
    overall_obj = json.load(f)
    for annotation in overall_obj:
      if annotation["fields"]["annotator_initials"] in VALID_ANNOTATORS:
        annotation_map[annotation["fields"]["review_id"]].append(annotation)

  overall_annotation_builder = {}
  for review_id, annotations in annotation_map.items():
    overall_annotation_builder[review_id] = Annotation(annotations)


if __name__ == "__main__":
  main()
