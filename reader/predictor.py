import json
from typing import TypedDict, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import QuestionAnsweringModelOutput

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
        if all(len(i["context"]) <= 0 for i in _input):
            return {"answer": self.nonce, "score_start": 1.0, "score_end": 1.0}
        inputs = [self.tk.tokenize(i) for i in _input]
        inputs_ids = self.to_model_input(inputs)
        outputs = self.model(**inputs_ids)
        answers = self.extract_answers(inputs, outputs)
        if all(ans["answer"] == "" for ans in answers):
            return {"answer": self.nonce, "score_start": 1.0, "score_end": 1.0}
        print(json.dumps(answers))
        return next(
            iter(sorted(answers, key=lambda x: x["score_start"] + x["score_end"]))
        )


if __name__ == "__main__":
    model_checkpoint = "./model"
    predictor = Predictor(model_checkpoint)

    _input: QuestionContextInput = {
        "question": "Thủ đô của Việt Nam?",
        "context": (
            "Hà Nội là thủ đô của nước Việt Nam và cũng là kinh đô của hầu hết các vương triều Việt trước đây. "
            "Do đó, lịch sử Hà Nội gắn liền với sự thăng trầm của lịch sử Việt Nam qua các thời kỳ. "
            "Hà Nội là thành phố lớn nhất Việt Nam về diện tích với 3328,9 km sau đợt mở rộng hành chính năm 2008, đồng thời cũng là địa phương đứng thứ nhì về dân số với 7.500.000 người (năm 2015). "
            "Hiện nay, thủ đô Hà Nội và thành phố Hồ Chí Minh là Đô thị Việt Nam của Việt Nam.Hà Nội nằm giữa đồng bằng sông Hồng trù phú, nơi đây đã sớm trở thành 1 trung tâm chính trị, kinh tế và văn hóa ngay từ những buổi đầu của lịch sử Việt Nam. "
        ),
    }
    answer = predictor.answer([_input])
    print(f"""Q: {_input["question"]}""")
    print(f"""A: {answer["answer"]}""")
