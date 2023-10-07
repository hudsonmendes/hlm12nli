# Python Built-in Modules
from typing import Dict, List


class Hlm12NliTokeniserSplitter:
    """
    Splits the texts into tokens, using the wordpiece[1] algorithm.

    Attributes:
        vocab: The vocabulary to use.
        expr_subtoken: The expression for the subtoken, default is "##".

    References:
        [1] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun,
        Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser,
        Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang,
        Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and
        Jeffrey Dean. 2016. Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine
        Translation. CoRR abs/1609.08144, (2016). Retrieved from http://arxiv.org/abs/1609.08144
    """

    vocab: Dict[str, int]
    expr_subtoken: str
    token_oov: str

    def __init__(
        self,
        vocab: Dict[str, int],
        token_oov: str,
        expr_subtoken: str,
    ) -> None:
        """
        Constructs a new Hlm12NliTokeniserSplitter.

        Args:
            vocab: The vocabulary to use.
            expr_subtoken: The expression for the subtoken, default is "##".
            token_oov: The out-of-vocabulary token, default is "[OOV]".
        """
        self.vocab = vocab
        self.expr_subtoken = expr_subtoken
        self.token_oov = token_oov

    def __call__(self, x: List[str]) -> List[str]:
        """
        Tokenizes the given texts, first splitting them by whitespace and then using the wordpiece algorithm.

        Args:
            x: The texts to tokenize.

        Returns:
            The tokenized texts.
        """
        y = [self._tokenize_by_whitespace(xi) for xi in x]
        y = [self._wordpiece(yi) for yi in y]
        return y

    def _tokenize_by_whitespace(self, text: str) -> List[str]:
        return text.strip().split() if text else []

    def _wordpiece(self, tokens: List[str]) -> List[str]:
        output_tokens = []
        for token in tokens:
            chars = list(token)
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = self.expr_subtoken + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.token_oov)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
