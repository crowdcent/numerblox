
from numerblox.key import Key, load_key_from_json

def test_key():
    pub_id, secret_key = "Hello", "World"
    example_key = Key(pub_id=pub_id, secret_key=secret_key)
    assert (example_key.pub_id, example_key.secret_key) == (pub_id, secret_key)

def test_load_key_from_json():
    example_key = load_key_from_json("tests/test_assets/mock_credentials.json")
    assert (example_key.pub_id, example_key.secret_key) == ("Hello", "World")
