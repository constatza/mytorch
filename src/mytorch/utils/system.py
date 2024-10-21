import importlib


def import_class(module_path: str, prepend: str = ""):
    """Import a module from a string path."""
    module_path = module_path.replace("/", ".").replace("\\", ".")
    split_path = [prepend] + module_path.split(".")
    module = importlib.import_module(".".join(split_path[:-1]))
    module_class = getattr(module, split_path[-1])
    return module_class
