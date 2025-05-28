
class StopwordDetector:
    def __init__(self, extractor, vowels: set = None, vowels_and_accented_vowels: set = None):
        self.extractor = extractor
        self.max_val = 99999999999
        self.delta_x = 5
        self.max_x_stopwords = 4000
        self.fake_sil_size = 15
        self.vowels = vowels
        self.vowels_and_accented_vowels = vowels_and_accented_vowels

    def detect_stopwords(self):
        # 1. Extract unigram frequencies
        stopword_candidates = {}
        for ngram, data in self.extractor.n_grams.items():
            if len(ngram) == 1:
                word = ngram[0]
                stopword_candidates[word] = data.frequency

        # 2. Compute frequency per syllable
        for word in stopword_candidates:
            syllables = self.calculate_syllables(word)
            if syllables == 0:
                syllables = self.fake_sil_size
            stopword_candidates[word] = stopword_candidates[word] / syllables

        # 3. Rank and select using elbow method
        ordered = dict(sorted(stopword_candidates.items(), key=lambda item: item[1], reverse=True))
        return self._select_stopwords(ordered)

    def _select_stopwords(self, ranked_dict):
        c = 0
        ant = self.max_val
        elbow_found = False
        stopwords = {}
        for word, value in ranked_dict.items():
            if c >= self.max_x_stopwords:
                break
            if not elbow_found and c % self.delta_x == 0:
                if ant - value < self.delta_x:
                    elbow_found = True
                ant = value
            if not elbow_found:
                stopwords[word] = True
            c += 1

        return set(stopwords.keys())
    
    def calculate_syllables(self, word: str):
        """
        Calculates the number of syllables in the word based on the presence of vowels and accented vowels.
        
        Args:
            vowels (set): Set of vowel characters.
            accented_vowels (set): Set of accented vowel characters.
        """
        n_vowels = 0
        n_vowels_before_accent = 0
        for i in range(len(word)):
            if word[i] in self.vowels_and_accented_vowels:
                n_vowels += 1
                if i < len(word)- 1 and word[i<+1] in self.vowels:
                    n_vowels_before_accent += 1

        return n_vowels - n_vowels_before_accent