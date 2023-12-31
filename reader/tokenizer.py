from typing import NamedTuple, TypedDict, Union

from underthesea import sent_tokenize, word_tokenize
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class QuestionContext(TypedDict):
    question: str
    context: str


def convert_tokens_to_ids(
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], word: str
) -> list[int]:
    result = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
    return [result] if isinstance(result, int) else result


class TokenizeResult(TypedDict):
    input_ids: list[int]
    words_lengths: list[int]
    valid: bool


class TokenizerWrapper(NamedTuple):
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

    def chunked_by_sentence(self, text: str, sentence_count=10, max_tokens=500, skip=0):
        assert 0 <= skip < sentence_count
        tokenizer = self.tokenizer
        sentences = sent_tokenize(text)
        sentence_tokens = [
            sum(len(tokenizer.encode(w)) for w in word_tokenize(sentence))
            for sentence in sentences
        ]
        chunk_sentences = list[str]()
        chunk_count = len(sentences) - sentence_count
        chunk_count = chunk_count if chunk_count > 0 else 1
        step = skip + 1
        for i in range(0, chunk_count, step):
            chunk_tokens = 0
            chunk_sentences.clear()
            last_chunk_index = min((i + sentence_count, len(sentences)))
            for j in range(i, last_chunk_index):
                token_count = sentence_tokens[j]
                if token_count + chunk_tokens > max_tokens:
                    break
                chunk_sentences.append(sentences[j])
                chunk_tokens += token_count
            yield " ".join(chunk_sentences)

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
