import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

def render_template(
    template: str,
    allow_missing: bool = False,
    skip_fields: list[str] | None = None,
    **kwargs: Any,
) -> str:

    from jinja2 import Template

    # .format and f-strings are quite different. .format doesn't support computation
    # in the curly braces. f-strings do, but f-string's evaluation cannot be delayed.
    # so for that i use Template(...). Template uses {{}} and format uses {}.
    skip_fields = skip_fields or []
    if skip_fields and allow_missing:
        raise ValueError("skip_fields and allow_missing should be used together.")
    for field in skip_fields:
        if field in kwargs:
            raise ValueError(
                f"skip_fields cannot contain fields that are in kwargs. Found: {field}"
            )
        kwargs[field] = ""
    result = Template(template).render(**kwargs)
    if not allow_missing:
        result = result.format(**kwargs)
        result = result.replace("__CURLY_OPEN__", "{").replace("__CURLY_CLOSE__", "}")
        return result
    kwargs_ = defaultdict(lambda: "", kwargs)
    result = result.format_map(kwargs_)
    result = result.replace("__CURLY_OPEN__", "{").replace("__CURLY_CLOSE__", "}")
    return result


def remove_empty_lines(text: str) -> str:
    lines = text.splitlines()
    stripped_lines = [line for line in lines if line.strip()]
    return "\n".join(stripped_lines)
