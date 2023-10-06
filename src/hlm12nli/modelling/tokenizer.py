# Python Built-in Modules
import json
import pathlib
from typing import Dict, List


class Hlm12NliTokenizer:
    """
    Construct a Hlm12Nli tokenizer, implemented using the wordpiece[1] algorithm.

    :param vocab_file: File containing the vocabulary.
    :param do_lower_case: Whether or not to lowercase the input when tokenizing.
    :param unk_token: The token that replaces any other that is out-of-vocabulary.
    :param pad_token: The token that is used for padding.
    :param cls_token: The token that is used for the start of a sequence.

    References:
        [1] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun,
        Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser,
        Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang,
        Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and
        Jeffrey Dean. 2016. Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine
        Translation. CoRR abs/1609.08144, (2016). Retrieved from http://arxiv.org/abs/1609.08144
    """

    vocab: Dict[str, int]
    do_lower_case: bool
    token_unk: str
    token_pad: str
    token_cls: str

    def __init__(
        self,
        vocab_file: pathlib.Path,
        do_lower_case=True,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        **kwargs,
    ):
        self.vocab = self.load_vocab(vocab_file)
        self.do_lower_case = do_lower_case

    @staticmethod
    def load_vocab(filepath: pathlib.Path) -> Dict[str, int]:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def tokenize(self, text, split_special_tokens=False):
        tokens = self._tokenize_by_whitespace(text)
        return self._wordpiece(tokens)

    def _tokenize_by_whitespace(self, text: str) -> List[str]:
        return text.strip().split() if text else []

    def _wordpiece(self, tokens: List[str]) -> List[str]:
        output_tokens = []
        for token in tokens:
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
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
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
