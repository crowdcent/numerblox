import json

class AttrDict(dict):
    """ Access dictionary elements as attributes. """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Key:
    """Numerai credentials."""
    def __init__(self, pub_id: str, secret_key: str):
        self.pub_id = pub_id
        self.secret_key = secret_key

    def __repr__(self):
        return f"Numerai Auth Key. pub_id = '{self.pub_id}'"

    def __str__(self):
        return self.__repr__()


def load_key_from_json(file_path: str, *args, **kwargs):
    """
    Initialize Key object from JSON file. \n
    Credentials file must have the following format: \n
    `{"pub_id": "PUBLIC_ID", "secret_key": "SECRET_KEY"}`
    """
    with open(file_path) as json_file:
        json_data = json.load(json_file, *args, **kwargs)
    pub_id = json_data["pub_id"]
    secret_key = json_data["secret_key"]
    return Key(pub_id=pub_id, secret_key=secret_key)
