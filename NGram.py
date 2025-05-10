from tqdm import tqdm

class NGram:
    def __init__(self, n, words: tuple):
        self.n = n
        self.words = words
        self.frequency = 0
        self.cohesion = 0
    
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


    def __str__(self) -> str:
        """
        Returns a string representation of the n-gram object.
        """
        return f"NGram(n = {self.n}, words = {self.words}, frequency={self.frequency}, cohesion={self.cohesion})"
