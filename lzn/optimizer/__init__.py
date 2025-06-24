import importlib


def from_config(model_groups, config):
    class_name = config.name
    params = config.params
    module_name, class_name = class_name.rsplit(".", 1)

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    param_groups = []
    for model, lr in model_groups:
        if model is not None:
            param_groups.append({"params": model.parameters(), "lr": lr})

    return class_(param_groups, **params)


def lr_scheduler_from_config(optimizer, config):
    class_name = config.name
    params = config.params
    module_name, class_name = class_name.rsplit(".", 1)

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_(optimizer, **params)
