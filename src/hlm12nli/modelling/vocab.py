# Python Built-in Modules
import json
import pathlib
from typing import Dict


class Hlm12NliTokenizerVocabReader:
    """
    Reads the vocabulary from a JSON file, postprocessing it
    should it require lowercasing.

    Attributes:
        filepath: The path to the JSON file containing the vocabulary.
    """

    filepath: pathlib.Path

    def __init__(self, filepath: pathlib.Path) -> None:
        """
        Constructs a new Hlm12NliTokenizerVocabReader.

        Args:
            filepath: The path to the JSON file containing the vocabulary.
        """
        self.filepath = filepath

    def read(self, do_lowercase: bool) -> Dict[str, int]:
        """
        Loads the vocabulary from the JSON file, postprocessing it
        should it require lowercasing.
        """
        raw = self._load_raw(self.filepath)
        return self._postprocess(raw, do_lowercase=do_lowercase)

    @staticmethod
    def _load_raw(filepath: pathlib.Path) -> Dict[str, int]:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _postprocess(raw: Dict[str, int], do_lowercase: bool) -> Dict[str, int]:
        out = {}
        if do_lowercase:
            for token, index in raw.items():
                lower_token = token.lower()
                if lower_token not in out:
                    out[lower_token] = index
                out[lower_token] = index
        return out
