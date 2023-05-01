"""Progress bars."""

import tqdm


def progressbar(x, /):
    return tqdm.tqdm(x)
