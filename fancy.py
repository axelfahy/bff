# -*- coding: utf-8 -*-
from typing import Sequence


def sliding_window(sequence: Sequence, window_size: int, step: int):
    """
    Apply a sliding window over the sequence.

    Each window is yielded. If there is a remainder, the remainder is yielded last,
    and will be smaller than the other windows.

    Parameters
    ----------
    sequence: Sequence
        Sequence to apply the sliding window on (can be str, list, numpy.array, etc.).
    window_size: int
        Size of the window to apply on the sequence.
    step: int
        Step for each sliding window.

    Yields
    ------
    Sequence
        Sequence generated.

    Examples
    --------
    >>> list(sliding_window('abcdef', 2, 1))
    ['ab', 'bc', 'cd', 'de', 'ef']
    >>> list(sliding_window(np.array([1, 2, 3, 4, 5, 6]), 5, 5))
    [array([1, 2, 3, 4, 5]), array([6])]
    """
    assert window_size >= step, 'Error: window_size must be larger or equal than step.'
    assert len(sequence) >= window_size, 'Error: length of sequence must be larger or equal than window_size.'
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception('Error: sequence must by iterable.')

    nb_chunks = int(((len(sequence) - window_size) / step) + 1)
    mod = window_size - nb_chunks
    for i in range(0, nb_chunks * step, step):
        yield sequence[i:i+window_size]
    if mod:
        yield sequence[len(sequence)-mod:]

