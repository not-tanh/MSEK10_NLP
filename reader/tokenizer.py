from typing import NamedTuple, TypedDict, Union

from underthesea import word_tokenize
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class QuestionContext(TypedDict):
    question: str
    context: str


def convert_tokens_to_ids(
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], word: str
):
    result = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
    return [result] if isinstance(result, int) else result


class TokenizeResult(TypedDict):
    input_ids: list[int]
    words_lengths: list[int]
    valid: bool


class TokenizerWrapper(NamedTuple):
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

    def tokenize(self, _input: QuestionContext) -> TokenizeResult:
        tokenizer = self.tokenizer
        question_word = word_tokenize(_input["question"])
        context_word = word_tokenize(_input["context"])

        question_sub_words_ids = [
            convert_tokens_to_ids(tokenizer, w) for w in question_word
        ]
        context_sub_words_ids = [
            convert_tokens_to_ids(tokenizer, w) for w in context_word
        ]
        valid = True
        observed_word_ids = [
            j for i in question_sub_words_ids + context_sub_words_ids for j in i
        ]
        if len(observed_word_ids) > tokenizer.max_len_single_sentence - 1:
            valid = False

        assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
        question_sub_words_ids = (
            [[tokenizer.bos_token_id]]
            + question_sub_words_ids
            + [[tokenizer.eos_token_id]]
        )
        context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

        input_ids = [
            j for i in question_sub_words_ids + context_sub_words_ids for j in i
        ]
        if len(input_ids) > tokenizer.max_len_single_sentence + 2:
            valid = False

        words_lengths = [
            len(item) for item in question_sub_words_ids + context_sub_words_ids
        ]

        return {"input_ids": input_ids, "words_lengths": words_lengths, "valid": valid}
