from numerblox.misc import AttrDict

def test_attrdict():
    test_dict = AttrDict({"test1": "hello", "test2": "world"})
    assert test_dict.test1 == test_dict['test1']
    assert test_dict.test2 == test_dict['test2']
