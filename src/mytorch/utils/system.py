import importlib
import inspect


def import_dynamically(module_path: str, prepend: str = ""):
    """
    Dynamically import a module, class, function, or attribute from a string path.

    Args:
        module_path (str): The string path of the module, class, function, or attribute to import.
        prepend (str, optional): Optional string to prepend to the module path. Defaults to "".

    Returns:
        The imported module, class, function, or attribute.
    """
    # Replace potential path separators with dots
    module_path = module_path.replace("/", ".").replace("\\", ".")

    # Prepend optional path, if provided
    if prepend:
        module_path = f"{prepend}.{module_path}"

    # Split the path into components
    path_parts = module_path.split(".")

    module = None
    # Try to progressively import the module components
    for i in range(1, len(path_parts)):
        try:
            # Try importing progressively larger parts of the path
            module = importlib.import_module(".".join(path_parts[:i]))
        except ModuleNotFoundError:
            # If any part of the module cannot be imported, return the error
            print(f"Cannot find module: {'.'.join(path_parts[:i])}")

    # If the last part is not found, try importing as an attribute (class, function, etc.)
    try:
        attr = getattr(module, path_parts[-1])
        return attr
    except AttributeError:
        # If it's not an attribute, return the full module instead
        return module


def filter_kwargs(cls, kwargs: dict):
    """
    Filter keyword arguments to only include valid parameters for a class constructor
    and return a new instance of the class with the filtered keyword arguments.
    :param cls: Class whose constructor parameters are used for filtering
    :param kwargs: Keyword arguments to filter
    :return: dict: Filtered keyword arguments
    """
    sig = inspect.signature(cls.__init__)
    # Get valid argument names (excluding 'self')
    valid_params = set(sig.parameters) - {"self"}

    # Filter kwargs to only include valid parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered_kwargs)
