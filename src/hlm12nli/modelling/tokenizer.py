# Python Built-in Modules
from abc import ABC
from typing import Dict, List


class Hlm12NliTokenizer:
    """
    Tokenizer for the Hlm12Nli model, implemented using the wordpiece[1] algorithm
    with padding, truncating and special tokens capabilities.

    References:
        [1] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun,
        Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser,
        Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang,
        Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and
        Jeffrey Dean. 2016. Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine
        Translation. CoRR abs/1609.08144, (2016). Retrieved from http://arxiv.org/abs/1609.08144
    """

    vocab: Dict[str, int]
    do_lowercase: bool
    special_tokens: List[str]
    seq_len: int | None = None
    max_seq_len: int = 1024
    token_oov: str
    token_pad: str
    token_str: str
    token_end: str

    def __init__(
        self,
        vocab: Dict[str, int],
        do_lowercase=True,
        seq_len=int | str,
        max_seq_len=1024,
        oov_token="[OOV]",
        pad_token="[PAD]",
        str_token="[STR]",
        end_token="[END]",
    ):
        """
        Constructs a new Hlm12NliTokenizer.

        Args:
            vocab: The vocabulary to use.
            do_lowercase: Whether to lowercase the tokens, default is True.
            oov_token: The out-of-vocabulary token, default is "[OOV]".
            pad_token: The padding token, default is "[PAD]".
            str_token: The start-of-sentence token, default is "[STR]".
            end_token: The end-of-sentence token, default is "[END]".
            seq_len: The fixed length of the sequence, None results in padding to max len in the batch, default is None.
            max_seq_len: The maximum sequence length, after which data will be truncated, default is 1024.
        """
        self.do_lowercase = do_lowercase
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.token_oov = oov_token
        self.token_pad = pad_token
        self.token_str = str_token
        self.token_end = end_token
        self.special_tokens = [self.token_oov, self.token_pad, self.token_str, self.token_end]
        self.vocab = vocab
        self._ensure_seq_length_constraints(self.seq_len, self.max_seq_len)
        self._ensure_respects_casing(self.vocab, do_lowercase=self.do_lowercase, ignore=self.special_tokens)
        self._ensure_special_tokens_in_vocab(self.vocab, special_tokens=self.special_tokens)

    @staticmethod
    def _ensure_seq_length_constraints(seq_len: int | str, max_seq_len: int) -> None:
        if seq_len is not None and seq_len > max_seq_len:
            raise Hlm12NliTokenizerSeqLenError(seq_len=seq_len, max_seq_len=max_seq_len)

    @staticmethod
    def _ensure_respects_casing(vocab: Dict[str, int], do_lowercase: bool, ignore: List[str]) -> None:
        if do_lowercase:
            for token in vocab.keys():
                if token not in ignore and token != token.lower():
                    raise Hlm12NliTokenizerVocabTokenCasingError(token=token)

    @staticmethod
    def _ensure_special_tokens_in_vocab(vocab: Dict[str, int], special_tokens: List[str]) -> None:
        for token in special_tokens:
            if token not in vocab:
                raise Hlm12NliTokenizerMissingSpecialTokenInVocab(token=token)

    def tokenize(self, x: List[str] | str) -> List[str]:
        single = False
        if not isinstance(x, list):
            x = [x]
            single = True
        y = [self._tokenize_by_whitespace(xi) for xi in x]
        y = [self._wordpiece(yi) for yi in y]
        y = [self._envelop(yi) for yi in y]
        y = self._pad(y)
        return y[0] if single else y

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
                output_tokens.append(self.token_oov)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def _envelop(self, tokens: List[str]) -> List[str]:
        return [self.token_str] + tokens + [self.token_end]

    def _pad(self, batch: List[List[str]]) -> List[str]:
        seq_len = self.seq_len or max(len(tokens) for tokens in batch)
        return [tokens + [self.token_pad] * (seq_len - len(tokens)) for tokens in batch]


class Hlm12NliTokenizerError(Exception, ABC):
    pass


class Hlm12NliTokenizerSeqLenError(Hlm12NliTokenizerError):
    def __init__(self, seq_len: int, max_seq_len: int):
        super().__init__(f"The sequence length {seq_len} is greater than the maximum sequence length {max_seq_len}.")


class Hlm12NliTokenizerVocabTokenCasingError(Hlm12NliTokenizerError):
    def __init__(self, token: str):
        super().__init__(f"The token '{token}' is not lowercase, but the tokenizer is set to lowercase tokens.")


class Hlm12NliTokenizerMissingSpecialTokenInVocab(Hlm12NliTokenizerError):
    def __init__(self, token: str):
        super().__init__(f"The special token '{token}' is not in the vocabulary.")
