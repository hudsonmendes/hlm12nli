# Python Built-in Modules
from copy import deepcopy
from typing import List


class Hlm12NliTokeniserJoiner:
    """
    Joins subtokens to tokens and then joins tokens to a text.
    """

    expr_subword: str

    def __init__(self, expr_subword: str) -> None:
        """
        Constructs a new instance of Hlm12NliTokeniserJoiner.

        Args:
            expr_subword: The subword to join to the previous token.
        """
        self.expr_subword = expr_subword

    def __call__(self, x: List[List[str]]) -> List[str]:
        """
        First glue subtokens to their original tokens, the joins them separating
        them by whitespace.

        Args:
            x: The subtokens to join.

        Returns:
            The sentence representing the joined tokens, with or without special tokens.
        """
        y = self._glue_subtokens_to_tokens(x)
        y = self._join_tokens(y)
        return y

    def _glue_subtokens_to_tokens(self, x: List[List[str]]) -> List[str]:
        expr_len = len(self.expr_subword)
        y = deepcopy(x)
        for y_index in range(len(y)):
            seq = y[y_index]
            for seq_index in range(len(seq)):
                i = len(x) - seq_index - 1
                token = seq[i]
                if token.startswith(self.expr_subword):
                    y[y_index][i - 1] += y[y_index][i][expr_len:]
                    del y[y_index][i]
            return y

    def _join_tokens(self, x: List[List[str]]) -> List[str]:
        return [" ".join(xi) for xi in x]
