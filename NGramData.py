from tqdm import tqdm


class NGramData:
    def __init__(self):
        self.frequency = 0
        self.scp = 0
        self.dice = 0
        self.phi_square = 0


    def __str__(self) -> str:
        """
        Returns a string representation of the n-gram object.
        """
        return f"frequency={self.frequency}, scp={self.scp}, dice={self.dice}, phi_square={self.phi_square}"
