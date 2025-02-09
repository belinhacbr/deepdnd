from deepdnd.app import ask_question

def test_ask_question():
    assert ask_question('How many days in a year?').startswith("<think>")