import string


def contains_alphanum(value: string):
    for char in value:
        if char.isalnum():
            return True
    return False

def contains_nonascii(value: string):
    for char in value:
        if not char.isascii():
            return True
    return False

if __name__ == '__main__':
    file = "train2"
    questions_file = open(f"output_files/{file}.from", buffering=1000, encoding="utf-8")
    answers_file = open(f"output_files/{file}.to", buffering=1000, encoding="utf-8")
    good_questions = []
    good_answers = []
    for (question, answer) in zip(questions_file, answers_file):
        if not contains_alphanum(answer) or contains_nonascii(answer) or contains_nonascii(question):
            continue
        tokens = answer.split(" ")
        tokens = list(filter(lambda x: x != '' and x != '\n', tokens))
        if len(tokens) >= 2:
            good_questions.append(question)
            good_answers.append(answer)


    with open(f"output_files/{file}_good.from", mode="a", encoding="utf-8") as good_questions_file:
        for good_question in good_questions:
            good_questions_file.write(good_question)
    with open(f"output_files/{file}_good.to", mode="a", encoding="utf-8") as good_answers_file:
        for good_answer in good_answers:
            good_answers_file.write(good_answer)