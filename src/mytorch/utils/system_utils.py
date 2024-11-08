import importlib
import os
import signal
import subprocess
import sys
import socket
import atexit


from mytorch.io.logging import get_logger

logger = get_logger(__name__)


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


def filter_kwargs(kwargs: dict):
    """
    Filter keyword arguments to only include valid parameters for a class constructor
    and return a new instance of the class with the filtered keyword arguments.
    """
    # sig = inspect.signature(cls.__init__)
    # # Get valid argument names (excluding 'self')
    # valid_params = set(sig.parameters) - {"self"}

    # Filter kwargs to only include valid parameters
    return {k: v for k, v in kwargs.items() if k != "name"}


def terminate_process_tree(pid):
    """
    Terminates a process and its child processes.

    Args:
        pid (int): The PID of the parent process.
    """
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(pid), signal.SIGTERM)  # Terminate process group
            logger.info(f"Terminated process group for PID: {pid}")
        elif os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=True)
            logger.info(f"Terminated process tree for PID: {pid}")
    except Exception as e:
        logger.error(f"Error terminating process tree for PID {pid}: {e}")


def check_port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.bind((host, port))
        except socket.error:
            logger.error(f"Port {port} is already in use.")
            sys.exit(1)


def setup_signal_handlers(process):
    """
    Sets up signal handlers to ensure graceful termination of the server process.

    Args:
        process (subprocess.Popen): The MLflow server process.
    """

    def cleanup(signum: int, frame):
        logger.info("Terminating MLflow server...")
        terminate_process_tree(process.pid)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, cleanup)  # Handle termination signals

    # Ensure cleanup on program exit
    atexit.register(cleanup)
