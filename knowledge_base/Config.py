import os
import json


class Config:
    def __init__(self, parameters_path='parameters.json'):
        self.parameters_path = parameters_path
        self._load()

    def _load(self):
        with open(self.parameters_path, 'r', encoding='utf-8') as f:
            params = json.load(f)

        self._apply(params)

    def _apply(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'parameters_path'}

    def update(self, para_dict):
        for key, value in para_dict.items():
            if not hasattr(self, key):
                raise KeyError(f'Parameter not found: {key}')
            setattr(key, value)
        self._validate_paths()
        self._save()


    def _validate_paths(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file does not exist: {self.data_path}")
        os.makedirs(self.model_document, exist_ok=True)

    def _save(self):
        params = self.as_dict()
        with open(self.parameters_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4)


if __name__ == '__main__':
    config = Config()
    params = config.as_dict()
    for k, v in params.items():
        print(k, v)
