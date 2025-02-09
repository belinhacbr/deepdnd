from deepdnd.model import prompt

def test_prompt():
    assert prompt().startswith("<think>")
