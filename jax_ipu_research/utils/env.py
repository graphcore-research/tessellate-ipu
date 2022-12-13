# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os


def env_append_path(envname: str, path: str):
    """Append an additional path to an existing env. variable. No-op if
    the path is already part of the env. variable.
    """
    env = os.environ.get(envname) or ""
    env_paths = set(env.split(":"))
    if path not in env_paths:
        os.environ[envname] = f"{env}:{path}"


def env_cpath_append(path: str):
    """Append a path to the CPATH env. variable."""
    return env_append_path("CPATH", path)


def env_path_append(path: str):
    """Append a path to the PATH env. variable."""
    return env_append_path("PATH", path)
