# Python Built-in Modules
from abc import ABC


class Hlm12NliTokeniserError(Exception, ABC):
    pass


class Hlm12NliTokeniserBatchRequiredError(Hlm12NliTokeniserError):
    def __init__(self):
        super().__init__("The tokeniser has been designed to work with batches, but a single input was given.")


class Hlm12NliTokeniserSeqLenError(Hlm12NliTokeniserError):
    def __init__(self, seq_len: int, max_seq_len: int):
        super().__init__(f"The sequence length {seq_len} is greater than the maximum sequence length {max_seq_len}.")


class Hlm12NliTokeniserVocabTokenCasingError(Hlm12NliTokeniserError):
    def __init__(self, token: str):
        super().__init__(f"The token '{token}' is not lowercase, but the tokeniser is set to lowercase tokens.")


class Hlm12NliTokeniserMissingSpecialTokenInVocabError(Hlm12NliTokeniserError):
    def __init__(self, token: str):
        super().__init__(f"The special token '{token}' is not in the vocabulary.")
