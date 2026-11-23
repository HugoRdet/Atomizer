import yaml
from pathlib import Path

def read_yaml(file_path: str | Path) -> dict:
    """
    Reads a YAML file from the given path and returns its content as a dictionary.

    Args:
        file_path (str | Path): The path to the YAML file.

    Returns:
        dict: The parsed YAML content.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"The YAML file was not found at: {path}")

    with open(path, "r", encoding="utf-8") as file:
        # safe_load prevents arbitrary code execution vulnerabilities
        return yaml.safe_load(file) or {}
