import errno
import io
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Text

# backwards compatibility 1.0.x
# noinspection PyUnresolvedReferences
from rasa.utils.io import read_json_file


def relative_normpath(f: Optional[Text], path: Text) -> Optional[Text]:
    """Return the path of file relative to `path`."""

    return os.path.normpath(os.path.relpath(f, path)) if f is not None else None


def lazyproperty(fn: Callable) -> Any:
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property."""

    attr_name = f"_lazy_{fn.__name__}"

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def list_to_str(l: List[Text], delim: Text = ", ", quote: Text = "'") -> Text:
    return delim.join([quote + e + quote for e in l])


def ordered(obj: Any) -> Any:
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    return sorted(ordered(x) for x in obj) if isinstance(obj, list) else obj


def module_path_from_object(o: Any) -> Text:
    """Returns the fully qualified class path of the instantiated object."""
    return f"{o.__class__.__module__}.{o.__class__.__name__}"


def json_to_string(obj: Any, **kwargs: Any) -> Text:
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def write_json_to_file(filename: Text, obj: Any, **kwargs: Any) -> None:
    """Write an object as a json string to a file."""

    write_to_file(filename, json_to_string(obj, **kwargs))


def write_to_file(filename: Text, text: Text) -> None:
    """Write a text to a file."""

    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(str(text))


def build_entity(
    start: int, end: int, value: Text, entity_type: Text, **kwargs: Dict[Text, Any]
) -> Dict[Text, Any]:
    """Builds a standard entity dictionary.

    Adds additional keyword parameters."""

    return {
        "start": start,
        "end": end,
        "value": value,
        "entity": entity_type,
    } | kwargs


def is_model_dir(model_dir: Text) -> bool:
    """Checks if the given directory contains a model and can be safely removed.

    specifically checks if the directory has no subdirectories and
    if all files have an appropriate ending."""
    allowed_extensions = {".json", ".pkl", ".dat"}
    dir_tree = list(os.walk(model_dir))
    if len(dir_tree) != 1:
        return False
    model_dir, child_dirs, files = dir_tree[0]
    file_extenstions = [os.path.splitext(f)[1] for f in files]
    return all(ext in allowed_extensions for ext in file_extenstions)


def is_url(resource_name: Text) -> bool:
    """Return True if string is an http, ftp, or file URL path.

    This implementation is the same as the one used by matplotlib"""

    URL_REGEX = re.compile(r"http://|https://|ftp://|file://|file:\\")
    return URL_REGEX.match(resource_name) is not None


def remove_model(model_dir: Text) -> bool:
    """Removes a model directory and all its content."""
    import shutil

    if not is_model_dir(model_dir):
        raise ValueError(
            f"Cannot remove {model_dir}, it seems it is not a model directory"
        )
    shutil.rmtree(model_dir)
    return True


def json_unpickle(file_name: Text) -> Any:
    """Unpickle an object from file using json."""
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    with open(file_name, "r", encoding="utf-8") as f:
        return jsonpickle.loads(f.read())


def json_pickle(file_name: Text, obj: Any) -> None:
    """Pickle an object to a file using json."""
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(jsonpickle.dumps(obj))
