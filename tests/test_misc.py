from numerblox.misc import AttrDict, Key, load_key_from_json

def test_attrdict():
    test_dict = AttrDict({"test1": "hello", "test2": "world"})
    assert hasattr(test_dict, 'test1')
    assert hasattr(test_dict, 'test2')
    assert test_dict.test1 == test_dict['test1']
    assert test_dict.test2 == test_dict['test2']

def test_key():
    pub_id, secret_key = "Hello", "World"
    example_key = Key(pub_id=pub_id, secret_key=secret_key)
    assert (example_key.pub_id, example_key.secret_key) == (pub_id, secret_key)

def test_load_key_from_json():
    example_key = load_key_from_json("tests/test_assets/mock_credentials.json")
    assert (example_key.pub_id, example_key.secret_key) == ("Hello", "World")
    