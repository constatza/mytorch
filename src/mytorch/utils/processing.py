from pathlib import Path
from urllib.parse import urlparse


def uri_to_path(uri: str) -> Path:
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "sqlite":
        raise ValueError(f"Invalid URI scheme: {parsed_uri.scheme}. Expected 'sqlite'.")

    path = parsed_uri.path.lstrip("/")
    if parsed_uri.netloc:  # Handle Windows drive letters
        path = f"{parsed_uri.netloc}:{path}"
    return Path(path)
