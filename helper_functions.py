import os
import re
from tqdm import tqdm


def load_and_preprocess_corpus(dir_path: str, special_chars: "list[str]" = None):
    """
    Loads and preprocesses text corpus from a directory of text files.
    Keeps URLs intact, spaces around punctuation, parentheses, and normalizes tokens.
    
    Args:
        dir_path (str): Path to directory with .txt files
        special_chars (list[str]): List of special characters to space (e.g., ['.', ',', ';', '!'])
    
    Returns:
        list[str]: Tokenized corpus
    """
    corpus = ""
    url_map = {}
    url_id = 0

    for file in tqdm(os.listdir(dir_path), unit="files", desc="Loading corpus"):
        with open(os.path.join(dir_path, file), "r", encoding="utf-8") as f:
            document = f.read()

            # Step 1: Detect URLs and replace them with placeholders
            url_pattern = r'https?://\S+|www\.\S+'
            urls_found = re.findall(url_pattern, document)
            for url in urls_found:
                placeholder = f'__URL_{url_id}__'
                url_map[placeholder] = url
                document = document.replace(url, placeholder)
                url_id += 1

            # Step 2: Add space after '(' and before ')'
            document = re.sub(r'\(\s*', '( ', document)
            document = re.sub(r'\s*\)', ' )', document)

            # Step 3: Space special characters (if any)
            if special_chars:
                pattern = r'(?<!\s)([' + re.escape(''.join(special_chars)) + r'])'
                document = re.sub(pattern, r' \1', document)

            # Step 4: Normalize newlines
            document = document.replace("\n", " \n ")

            # Step 5: Append cleaned document to full corpus
            corpus += document + " "

    # Step 6: Restore URL placeholders with actual URLs
    tokens = corpus.split()
    tokens = [url_map.get(token, token) for token in tokens if token not in {"", "\n"}]

    return tokens


def is_previous_substring_capitalized(corpus: str, index: int) -> bool:
    """
    Checks if the substring before the index is capitalized.
    
    Args:
        corpus (str): The text corpus to analyze.
        index (int): The index to check.
    Returns:
        bool: True if the substring before the index is capitalized, False otherwise.
    """
    # Check if the substring before the index is uppercase
    is_upper = False
    while index > 0 and corpus[index] != " ":
        if corpus[index].isupper():
            is_upper = True
        index -= 1
    return is_upper

def find_words_frequencies(corpus: str):
    """
    Finds the frequency of each word in the corpus, showing progress bar.
    
    Args:
        corpus (str): The text corpus to analyze.
    Returns:
        dict: A dictionary with words as keys and their frequencies as values.
    """

    word_counter = {}
    # Tokenize each sentence (based on whitespace) and update the frequency count
    for word in corpus:
        # ignore newline characters
        if word in word_counter.keys():
            word_counter[word] += 1
        else:
            word_counter[word] = 1

    # Sort the dictionary by frequency in descending order
    word_counter = dict(sorted(word_counter.items(), key=lambda item: item[1], reverse=True))
    return word_counter


