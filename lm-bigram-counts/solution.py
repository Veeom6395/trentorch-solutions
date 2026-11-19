
import numpy as np


def bigram_counts(words: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Returns (counts, itos). `itos` is ['.'] plus the sorted distinct
    characters of all words. counts[i, j] is how often itos[j] directly
    followed itos[i] when every word is wrapped as '.' + word + '.'.
    """

    # 1. Collect all unique characters and sort them
    itos = ['.'] + sorted(set(''.join(words)))

    # 2. Create a mapping from each character to its index
    stoi = {ch: i for i, ch in enumerate(itos)}

    # 3. Initialize a V x V matrix of zeros
    V = len(itos)
    counts = np.zeros((V, V), dtype=int)

    # 4. Wrap each word and count adjacent character pairs
    for word in words:
        wrapped = '.' + word + '.'

        for ch1, ch2 in zip(wrapped, wrapped[1:]):
            i = stoi[ch1]
            j = stoi[ch2]
            counts[i, j] += 1

    return counts, itos
