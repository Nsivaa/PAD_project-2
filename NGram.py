from tqdm import tqdm
from enum import Enum, auto


class NGram:
    def __init__(self):
        self.frequency = 0
        self.cohesion = 0

    class GluesEnum(Enum):
        Scp = auto()
        Dice = auto()
        Phi_square = auto()

    def calculate_scp(self, corpus_size: int):
        p_w = self.frequency / corpus_size
        if p_w == 0:
            return 0
        F = 0
        for i in range(1, self.n):
            left = self.words[:i]
            right = self.words[i:]
            p_left = find_ngram_freq(left) / corpus_size
            p_right = find_ngram_freq(right) / corpus_size
            F += p_left * p_right

        F /= (self.n - 1)

        # SCP-f = (p(w1...wn)^2) / F
        return (p_ngram ** 2) / F if F > 0 else 0
                

    def calculate_dice(self, corpus_size: int):
        pass

    def calculate_phi_square(self, corpus_size: int):
            pass

    GLUE_FUNCTIONS = {
        GluesEnum.Scp : calculate_scp,
        GluesEnum.Dice: calculate_dice,
        GluesEnum.Phi_square: calculate_phi_square
    }
    

    def find_frequency(self, corpus: str):
        """
        Finds the frequency of the n-gram in the corpus and assigns it to the frequency attribute.
        Params:
            corpus (str): The text corpus to analyze.
        """

        # Iterate over the words and count the frequency of the n-gram
        for i in range(len(corpus) - self.n + 1):
            n_gram = " ".join(self.words)
            # Check if the n-gram is present in the tokens
            if n_gram == " ".join(corpus[i:i + self.n]):
                self.frequency += 1

    def find_cohesion(self, corpus: str, type: str):
        """
        finds the cohesion of the n-gram in the corpus and assigns it to the cohesion attribute.
        """

    

    def __str__(self) -> str:
        """
        Returns a string representation of the n-gram object.
        """
        return f"frequency={self.frequency}, cohesion={self.cohesion})"
