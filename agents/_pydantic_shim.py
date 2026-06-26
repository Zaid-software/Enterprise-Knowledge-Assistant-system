import json
from dataclasses import dataclass, field, fields, asdict
from typing import get_type_hints


def Field(default=None, default_factory=None, **kwargs):
    if default_factory is not None:
        return field(default_factory=default_factory)
    return field(default=default)


class _BaseModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != "BaseModel":
            # kw_only avoids "non-default argument follows default argument"
            # errors when a model has required fields declared after optional
            # ones (pydantic allows this; vanilla dataclasses don't).
            cls = dataclass(cls, kw_only=True)
        return cls


class BaseModel(metaclass=_BaseModelMeta):
    def model_dump(self):
        return asdict(self)

    def dict(self):
        return asdict(self)

    def model_dump_json(self, indent=None):
        return json.dumps(self.model_dump(), indent=indent, default=str)

    def json(self, indent=None):
        return self.model_dump_json(indent=indent)
