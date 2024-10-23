__all__ = ["pretty_format_dict_str"]

import re


def pretty_format_dict_str(dict_str: str, indent: int = 1) -> str:
    """Cannot believe this is not built-in...."""
    dict_str = f"{dict_str}"
    # Add a new line after each comma and indent key-value pairs
    dict_str = re.sub(r",", ",\n    ", dict_str)
    # Add newline and indent after opening brace
    dict_str = re.sub(r"{", "{\n    ", dict_str, indent)
    # Add newline before closing brace
    dict_str = re.sub(r"}", "\n}", dict_str, indent)
    return dict_str
