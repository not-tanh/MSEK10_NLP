import json
import time
from typing import TypedDict, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from underthesea import sent_tokenize, word_tokenize

from .pretrained_model import MRCQuestionAnswering
from .tokenizer import TokenizeResult, TokenizerWrapper


class QuestionContextInput(TypedDict):
    question: str
    context: str


class Prediction(TypedDict):
    answer: str
    score_start: float
    score_end: float


def pad_tokens(
    values: list[torch.Tensor],
    pad_idx: Union[float, int],
    eos_idx: Union[float, int, None] = None,
    left_pad=False,
    move_eos_to_beginning=False,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""

    def copy_tensor(
        src: torch.Tensor,
        dst: torch.Tensor,
    ):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert isinstance(eos_idx, (float, int))
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    size = max(v.size(0) for v in values)
    padded = values[0].new(len(values), size).fill_(pad_idx)

    for i, v in enumerate(values):
        without_padding = (
            padded[i][size - len(v) :] if left_pad else padded[i][: len(v)]
        )
        copy_tensor(v, without_padding)
    return padded


def chunked_by_sentence(text: str, sentence_count=5, max_tokens=500, skip=0):
    assert 0 <= skip < sentence_count
    sentences = sent_tokenize(text)
    sentence_tokens = [len(word_tokenize(sentence)) for sentence in sentences]
    chunk_sentences = list[str]()
    chunk_count = len(sentences) - sentence_count
    step = skip + 1
    for i in range(0, chunk_count, step):
        chunk_tokens = 0
        chunk_sentences.clear()
        for j in range(i, i + sentence_count):
            token_count = sentence_tokens[j]
            if token_count + chunk_tokens > max_tokens:
                break
            chunk_sentences.append(sentences[j])
            chunk_tokens += token_count
        yield " ".join(chunk_sentences)


def chunk_input(_input: QuestionContextInput):
    for ctx_chunk in chunked_by_sentence(_input["context"]):
        yield QuestionContextInput(question=_input["question"], context=ctx_chunk)


class Predictor:
    nonce = "Không tìm được câu trả lời, bạn vui lòng Google nhé!"

    def __init__(self, model_checkpoint: str = "./model"):
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_checkpoint)  # type: ignore
        self.model: torch.nn.Module = MRCQuestionAnswering.from_pretrained(model_checkpoint)  # type: ignore
        self.tk = TokenizerWrapper(self.tokenizer)

    def to_model_input(self, samples: list[TokenizeResult]) -> dict[str, torch.Tensor]:
        if len(samples) == 0:
            return {}
        tokenizer = self.tokenizer
        assert tokenizer.pad_token_id is not None
        input_ids = pad_tokens(
            [torch.tensor(item["input_ids"]) for item in samples],
            pad_idx=tokenizer.pad_token_id,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(samples)):
            attention_mask[i][: len(samples[i]["input_ids"])] = 1
        words_lengths = pad_tokens(
            [torch.tensor(item["words_lengths"]) for item in samples], pad_idx=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_lengths": words_lengths,
        }

    def extract_answers(
        self,
        inputs: list[TokenizeResult],
        outputs: QuestionAnsweringModelOutput,
    ) -> list[Prediction]:
        plain_result = []
        tokenizer = self.tokenizer
        for sample_input, start_logit, end_logit in zip(
            inputs, outputs.start_logits, outputs.end_logits
        ):
            sample_words_length = sample_input["words_lengths"]
            input_ids = sample_input["input_ids"]
            # Get the most likely beginning of answer with the argmax of the score
            answer_start = sum(sample_words_length[: torch.argmax(start_logit)])
            # Get the most likely end of answer with the argmax of the score
            answer_end = sum(sample_words_length[: torch.argmax(end_logit) + 1])

            if answer_start <= answer_end:
                tokens = tokenizer.convert_ids_to_tokens(
                    input_ids[answer_start:answer_end]
                )
                answer = tokenizer.convert_tokens_to_string(
                    [tokens] if isinstance(tokens, str) else tokens
                )
                if answer == tokenizer.bos_token:
                    answer = ""
            else:
                answer = ""

            score_start = (
                torch.max(torch.softmax(start_logit, dim=-1))
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )
            score_end = (
                torch.max(torch.softmax(end_logit, dim=-1))
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )
            plain_result.append(
                {"answer": answer, "score_start": score_start, "score_end": score_end}
            )
        return plain_result

    def answer(self, _input: list[QuestionContextInput]):
        t = time.time()
        if all(len(i["context"]) <= 0 for i in _input):
            return {"answer": self.nonce, "score_start": 1.0, "score_end": 1.0}
        _input = [x for i in _input for x in chunk_input(i)]
        inputs = [self.tk.tokenize(i) for i in _input]
        inputs_ids = self.to_model_input(inputs)
        outputs = self.model(**inputs_ids)
        answers = self.extract_answers(inputs, outputs)
        if all(ans["answer"] == "" for ans in answers):
            return {"answer": self.nonce, "score_start": 1.0, "score_end": 1.0}
        # print(json.dumps(answers))
        print(f"Prediction time: {time.time() - t}")
        return next(
            iter(sorted(answers, key=lambda x: x["score_start"] + x["score_end"]))
        )
