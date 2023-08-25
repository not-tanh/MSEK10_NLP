from underthesea import word_tokenize
import string


class Preprocessor:
    def __init__(self, stopwords_path: str):
        self.stopwords = set()
        self.punct = set([c for c in string.punctuation]) | {'“', '”', "...", "…", "..", "•", '“', '”'}

        with open(stopwords_path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace(' ', '_')
                if line:
                    self.stopwords.add(line.strip())

    def clean_text(self, text: str) -> str:
        tokens = word_tokenize(text)
        res = []
        # Remove stopwords and
        for token in tokens:
            token = token.replace(' ', '_')
            if token.lower() not in self.stopwords and token not in self.punct:
                res.append(token)
        return ' '.join(res)


if __name__ == '__main__':
    text1 = "Cô gái 9X Quảng Trị khởi nghiệp từ nấm sò rằng thì là"
    text2 = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump thì là"
    # print(question_segmentation(question2))

    p = Preprocessor('../stopwords.txt')
    print(p.clean_text(text1))
    print(p.clean_text(text2))
