import retriever
from reader.predictor import Predictor, QuestionContextInput


def test_retriever():
    results = retriever.find_relevant_documents("thủ đô của Việt Nam là gì?")
    print(results)
    
    
def test_reader():
    model_checkpoint_path = "./model"
    predictor = Predictor(model_checkpoint_path)

    _input: QuestionContextInput = {
        "question": "Thủ đô của Việt Nam?",
        "context": (
            "Hà Nội là thủ đô của nước Việt Nam và cũng là kinh đô của hầu hết các vương triều Việt trước đây. "
            "Do đó, lịch sử Hà Nội gắn liền với sự thăng trầm của lịch sử Việt Nam qua các thời kỳ. "
            "Hà Nội là thành phố lớn nhất Việt Nam về diện tích với 3328,9 km sau đợt mở rộng hành chính năm 2008, "
            "đồng thời cũng là địa phương đứng thứ nhì về dân số với 7.500.000 người (năm 2015)."
            "Hiện nay, thủ đô Hà Nội và thành phố Hồ Chí Minh là Đô thị Việt Nam của Việt Nam.Hà Nội nằm giữa đồng "
            "bằng sông Hồng trù phú, nơi đây đã sớm trở thành 1 trung tâm chính trị, kinh tế và văn hóa ngay từ những "
            "buổi đầu của lịch sử Việt Nam."
        ),
    }
    _answer = predictor.answer([_input])
    print(f"""Q: {_input["question"]}""")
    print(f"""A: {_answer["answer"]}""")


if __name__ == '__main__':
    test_reader()
