# Copyright 2020 The HuggingFace Team. All rights reserved.
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
# Modified for FlowHOI; see third_party/licenses/transformers-Apache-2.0.txt.

import dataclasses
import sys
from argparse import ArgumentParser, ArgumentTypeError, SUPPRESS
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise ArgumentTypeError("Expected true or false.")


class HfArgumentParser(ArgumentParser):
    """Build argparse options from dataclass type hints and field metadata."""

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType,
                                              Iterable[DataClassType]],
                 **kwargs):
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released.")
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List, ):
                    if (typestring ==
                            f"typing.Union[{collection[prim_type]}, NoneType]"
                            or typestring
                            == f"typing.Optional[{collection[prim_type]}]"):
                        field.type = collection[prim_type]
                if (typestring
                        == f"typing.Union[{prim_type.__name__}, NoneType]"
                        or typestring
                        == f"typing.Optional[{prim_type.__name__}]"):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool or field.type is Optional[bool]:
                kwargs.update(type=_parse_bool, nargs="?", const=True,
                              default=field.default)
                self.add_argument(field_name, **kwargs)
                self.add_argument(f"--no_{field.name}", dest=field.name,
                                  action="store_false", default=SUPPRESS,
                                  help=f"Set {field.name} to false.")
                continue
            elif hasattr(field.type, "__origin__") and issubclass(
                    field.type.__origin__, List):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(x == kwargs["type"] for x in field.type.__args__
                           ), "{} cannot be a List of mixed types".format(
                               field.name)
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
            self,
            args=None,
            return_remaining_strings=False,
            look_for_args_file=True,
            args_filename=None) -> Tuple[DataClass, ...]:
        """Parse arguments into dataclass instances and optional remaining values."""
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[
                    1:]
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(
                    f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}"
                )

            return (*outputs, )
