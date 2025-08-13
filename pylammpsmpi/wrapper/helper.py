# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Any, List
from inspect import isfunction, getsource


def serialize_functions(args: List[Any]) -> List[Any]:
    """
    Replace any function objects in the argument list with a tuple containing a marker, the function's name, and its source code.

    Args:
        args (List[Any]): Arguments to check.

    Returns:
        List[Any]: Arguments with functions replaced by tuples.
    """
    out = []
    for arg in args:
        if isfunction(arg):
            out.append(("__FUNC__", arg.__name__, getsource(arg)))
        else:
            out.append(arg)
    return out


def deserialize_functions(args: List[Any]) -> List[Any]:
    """
    Restore any functions previously serialized with serialize_functions.

    Args:
        args (List[Any]): Arguments to check.

    Returns:
        List[Any]: Arguments with serialized functions restored.
    """
    out = []
    for arg in args:
        if isinstance(arg, tuple) and len(arg) == 3 and arg[0] == "__FUNC__":
            ns = {}
            exec(arg[2], ns)
            out.append(ns[arg[1]])
        else:
            out.append(arg)
    return out 