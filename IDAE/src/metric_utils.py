from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score


class _NoWordNet:
    @staticmethod
    def synsets(_word):
        return []


try:
    wordnet.ensure_loaded()
    METEOR_WORDNET = wordnet
except LookupError:
    METEOR_WORDNET = _NoWordNet()


def calculate_meteor(reference: str, hypothesis: str) -> float:
    return meteor_score(
        [str(reference).split()],
        str(hypothesis).split(),
        wordnet=METEOR_WORDNET,
    )
