import importlib


def from_config(config, **additional_params):
    class_name = config.name
    params = config.params
    module_name, class_name = class_name.rsplit(".", 1)

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_(**params, **additional_params)
