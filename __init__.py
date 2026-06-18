# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DiffDrive package.

This is a standalone directory (not nested inside a `robot/` package), so
all imports here and across the sibling modules are plain (e.g.
`from diffdrive_client import DiffDriveClient`) rather than relative
(`from .diffdrive_client import ...`). That means this folder's directory
must be on `sys.path` — true automatically if you run scripts from inside
it, or if you `sys.path.insert(0, "/path/to/kibub-diff-drive")` before
importing from elsewhere.

Imports are also lazy: importing this package does not require `pyserial`
(needed only by DiffDrive, the host-side motor driver) or `draccus` (needed
only by diffdrive_host's CLI entry point). This means a client-only machine
(e.g. a laptop that only ever runs DiffDriveRemote/DiffDriveClient over the
network) doesn't need those installed.
"""

import os
import sys

# Ensure sibling modules are importable even if this package is imported
# from outside the kibub-diff-drive directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_diffdrive import (
    DiffDriveClientConfig as DiffDriveClientConfig,
    DiffDriveConfig as DiffDriveConfig,
    DiffDriveHostConfig as DiffDriveHostConfig,
)
from diffdrive_client import DiffDriveClient as DiffDriveClient
from easy_diffdrive import (
    DiffDriveNotConnectedError as DiffDriveNotConnectedError,
    DiffDriveRemote as DiffDriveRemote,
    WheelState as WheelState,
)

_LAZY_HOST_ATTRS = {
    "DiffDrive": ("diffdrive", "DiffDrive"),
    "DiffDriveHost": ("diffdrive_host", "DiffDriveHost"),
    "DiffDriveServerConfig": ("diffdrive_host", "DiffDriveServerConfig"),
}


def __getattr__(name: str):
    # Loads host-only classes (and their pyserial/draccus deps) on first
    # access instead of at package-import time. See module docstring.
    if name in _LAZY_HOST_ATTRS:
        module_name, attr_name = _LAZY_HOST_ATTRS[name]
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache for next access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
