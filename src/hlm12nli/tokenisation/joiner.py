# Python Built-in Modules
from copy import deepcopy
from typing import List


class Hlm12NliTokeniserJoiner:
    """
    Joins subtokens to tokens and then joins tokens to a text.
    """

    special_tokens: List[str]
    expr_subword: str

    def __init__(self, special_tokens: List[str], expr_subword: str) -> None:
        self.special_tokens = special_tokens
        self.expr_subword = expr_subword

    def __call__(self, x: List[List[str]], ignore_special_tokens: bool = False) -> List[str]:
        """
        First glue subtokens to their original tokens, the joins them separating
        them by whitespace.

        Args:
            x: The subtokens to join.
            ignore_special_tokens: Whether to ignore special tokens, default is False.
        """
        y = self._glue_subtokens_to_tokens(x)
        if ignore_special_tokens:
            y = self._remove_special_tokens(y)
        y = self._join_tokens(y)
        return y

    def _glue_subtokens_to_tokens(self, x: List[List[str]]) -> List[str]:
        expr_len = len(self.expr_subword)
        y = deepcopy(x)
        for cursor in range(len(x)):
            i = len(x) - cursor - 1
            if x[i].startswith(self.expr_subword):
                y[i - 1] += x[i][expr_len:]
                del y[i]
        return y

    def _join_tokens(self, x: List[List[str]]) -> List[str]:
        return " ".join(x)

    def _remove_special_tokens(self, x: List[str]) -> List[str]:
        return [xi for xi in x if xi not in self.special_tokens]
