from pytest import fixture
from deepdnd.app import DeepDnd

@fixture
def test_app():
    deepdndtest = DeepDnd(version='test')
    # deepdndtest.index_documents()
    return deepdndtest


def test_ask_question(test_app):
    assert test_app.ask_question('How many days in a year?').startswith("<think>")