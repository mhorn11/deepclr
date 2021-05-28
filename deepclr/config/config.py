from __future__ import annotations

from collections import OrderedDict
import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import yaml


class ConfigEnum(Enum):
    """Enumerate for configuration data that can be created from a string."""
    @classmethod
    def create(cls, x: Union[str, ConfigEnum]) -> Any:
        if isinstance(x, str):
            return cls[x.upper()]
        elif isinstance(x, ConfigEnum):
            return x
        else:
            raise KeyError(f"Invalid config enum member '{x}'")


class ConfigParam:
    """Store a config parameter with additional information."""
    def __init__(self, default: Any = None, required: Optional[bool] = None, doc: Optional[str] = None,
                 internal: Optional[bool] = None):
        if required is None:
            required = False
        if doc is None:
            doc = ""
        if internal is None:
            internal = False

        if required:
            self.__value = None
            self.__valid = False
        else:
            self.__value = default
            self.__valid = True

        self.__default = default
        self.__required = required
        self.__doc = doc
        self.__internal = internal

    @property
    def value(self) -> Any:
        return self.__value

    @value.setter
    def value(self, v: Any) -> None:
        self.__value = v
        self.__valid = True

    @property
    def required(self) -> bool:
        return self.__required

    @property
    def doc(self) -> str:
        return self.__doc

    def __str__(self) -> str:
        return str(self.value)

    def is_valid(self) -> bool:
        return self.__valid

    def is_internal(self) -> bool:
        return self.__internal

    def reset(self) -> None:
        if self.__required:
            self.__value = None
            self.__valid = False
        else:
            self.__value = self.__default
            self.__valid = True


class ConfigParamGroup(OrderedDict):
    """Parameter group containing other groups or parameters."""
    ALLOW_DYNAMIC_PARAMS = '__allow_dynamic_params__'
    IMMUTABLE = '__immutable__'
    INTERNAL = '__internal__'

    def __init__(self, allow_dynamic_params: Optional[bool] = None, internal: Optional[bool] = None):
        if allow_dynamic_params is None:
            allow_dynamic_params = False
        if internal is None:
            internal = False
        super().__init__()
        self.__dict__[ConfigParamGroup.ALLOW_DYNAMIC_PARAMS] = allow_dynamic_params
        self.__dict__[ConfigParamGroup.IMMUTABLE] = False
        self.__dict__[ConfigParamGroup.INTERNAL] = internal

    def __str__(self):
        return str(self._value_dict(invalid=True, internal=True))

    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __getitem__(self, key: str) -> Any:
        if key in self:
            attr = super().__getitem__(key)
        else:
            raise AttributeError("Attribute '{}' does not exist".format(key))
        if isinstance(attr, ConfigParam):
            return attr.value
        else:
            return attr

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if self.is_frozen():
            raise AttributeError("Cannot change frozen config")
        if key in self:
            param = super().__getitem__(key)
        else:
            if self._allow_dynamic_params():
                param = self._add_param(key, internal=True)
            else:
                raise AttributeError("Parameter '{}' does not exist".format(key))
        if not isinstance(param, ConfigParam):
            raise AttributeError("Attribute '{}' is not a parameter".format(key))
        param.value = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def __delitem__(self, key: str) -> None:
        if self.is_frozen():
            raise AttributeError("Cannot change frozen config")
        if key not in self:
            raise AttributeError("Attribute '{}' does not exist".format(key))
        super().__delitem__(key)

    def __copy__(self) -> ConfigParamGroup:
        if self.is_frozen():
            raise AttributeError("Cannot copy frozen config. You can use deepcopy() instead.")
        new_config = ConfigParamGroup()
        new_config.__dict__.update(self.__dict__)
        for k, v in self.items():
            super(ConfigParamGroup, new_config).__setitem__(k, v)
        return new_config

    def __deepcopy__(self, memo: Dict) -> ConfigParamGroup:
        new_config = ConfigParamGroup()
        new_config.__dict__.update(self.__dict__)
        for k, v in self.items():
            super(ConfigParamGroup, new_config).__setitem__(k, copy.deepcopy(v, memo))
        return new_config

    def _add_group(self, name: str, parent: Optional[Union[ConfigParamGroup, List[str], str]] = None,
                   **kwargs: Any) -> ConfigParamGroup:
        if self.is_frozen():
            raise AttributeError("Cannot change frozen config")
        if self.is_internal():
            kwargs['internal'] = True
        if parent is None or (isinstance(parent, (list, str)) and len(parent) == 0):
            return self._add_attr(name, ConfigParamGroup(**kwargs))
        else:
            group, remaining = self.__get_parent(parent)
            return group._add_group(name, parent=remaining, **kwargs)

    def _add_param(self, name: str, parent: Optional[Union[ConfigParamGroup, List[str], str]] = None,
                   **kwargs: Any) -> ConfigParam:
        if self.is_frozen():
            raise AttributeError("Cannot change frozen config")
        if self.is_internal():
            kwargs['internal'] = True
        if parent is None or (isinstance(parent, (list, str)) and len(parent) == 0):
            return self._add_attr(name, ConfigParam(**kwargs))
        else:
            group, remaining = self.__get_parent(parent)
            return group._add_param(name, parent=remaining, **kwargs)

    def __get_parent(self, parent: Union[ConfigParamGroup, List[str], str]) -> \
            Tuple[ConfigParamGroup, Optional[List]]:
        if isinstance(parent, ConfigParamGroup):
            group = parent
            remaining = None
        else:
            if isinstance(parent, list):
                group_name = parent[0]
                remaining = parent[1:]
            else:
                group_name = parent
                remaining = None

            if group_name in self:
                group = self[group_name]
            else:
                if self._allow_dynamic_params():
                    group = self._add_group(group_name, internal=False)
                else:
                    raise AttributeError("Parent group '{}' does not exist".format(group_name))

        return group, remaining

    def _add_attr(self, name: str, value: Any) -> Any:
        if self.is_frozen():
            raise AttributeError("Cannot change frozen config")
        if name in self:
            raise AttributeError("Attribute '{}' already exists".format(name))
        super().__setitem__(name, value)
        return super().__getitem__(name)

    def _read_dict(self, data: Dict) -> None:
        if not isinstance(data, dict):
            return

        # add known attributes
        for name, attr in self.items():
            if name not in data:
                continue
            if isinstance(attr, ConfigParamGroup):
                attr._read_dict(data[name])
            else:
                self[name] = data[name]

        # check for new attributes
        for name, value in data.items():
            if name in self:
                continue
            if not self._allow_dynamic_params():
                warnings.warn("Attribute '{}' of input is ignored".format(name))
                continue
            if isinstance(value, dict):
                group = self._add_group(name, allow_dynamic_params=True, internal=True)
                group._read_dict(value)
            else:
                param = self._add_param(name, internal=True)
                param.value = value

    def _value_dict(self, invalid: bool = True, internal: bool = True) -> OrderedDict:
        values = OrderedDict(
            [(key,
              attr.value if isinstance(attr, ConfigParam)
              else attr._value_dict(invalid=invalid, internal=internal))
             for key, attr in self.items()
             if (invalid or attr.is_valid() or isinstance(attr, ConfigParamGroup)) and
                (internal or not attr.is_internal())]
        )
        return values

    def _immutable(self, is_immutable: bool) -> None:
        self.__dict__[ConfigParamGroup.IMMUTABLE] = is_immutable
        for name, attr in self.items():
            if isinstance(attr, ConfigParamGroup):
                attr._immutable(is_immutable)

    def _allow_dynamic_params(self) -> bool:
        return self.__dict__[ConfigParamGroup.ALLOW_DYNAMIC_PARAMS]

    def is_valid(self) -> bool:
        for attr in self.values():
            if not attr.is_valid():
                return False
        return True

    def is_internal(self) -> bool:
        return self.__dict__[ConfigParamGroup.INTERNAL]

    def freeze(self) -> None:
        self._immutable(True)

    def defrost(self) -> None:
        self._immutable(False)

    def is_frozen(self) -> bool:
        return self.__dict__[ConfigParamGroup.IMMUTABLE]

    def reset(self) -> None:
        if self.is_frozen():
            raise AttributeError("Cannot change frozen config")
        for name, attr in self.items():
            if attr.is_internal():
                del self[name]
            else:
                attr.reset()

    def dump(self, intend: int = 0) -> str:
        s = ""
        for i, (key, attr) in enumerate(self.items()):
            if i != 0:
                s += "\n"
            s += " " * intend + "{}:".format(key)
            if not isinstance(attr, ConfigParamGroup):
                s += " {}".format(attr.value)
                if attr.required:
                    s += " <required>"
            if not attr.is_valid():
                s += " <invalid>"
            if attr.is_internal():
                s += " <internal>"
            if isinstance(attr, ConfigParamGroup):
                s += "\n" + attr.dump(intend + 2)
        return s

    def copy(self, defrost: bool = True) -> Config:
        config_copy = copy.deepcopy(self)
        if defrost:
            config_copy.defrost()
        return config_copy

    def dict(self) -> OrderedDict:
        data = [(k, v.dict() if isinstance(v, ConfigParamGroup) else v.value)
                for k, v in self.items()]
        return OrderedDict(data)

    def define_group(self, name: str, parent: Optional[Union[ConfigParamGroup, List[str], str]] = None)\
            -> ConfigParamGroup:
        return self._add_group(name, parent=parent, allow_dynamic_params=self._allow_dynamic_params(), internal=False)

    def define_param(self, name: str, parent: Optional[Union[ConfigParamGroup, List[str], str]] = None,
                     default: Any = None, required: Optional[bool] = None, doc: Optional[str] = None) -> None:
        self._add_param(name, parent=parent, default=default, required=required, doc=doc, internal=False)

    def add_internal_group(self, name: str, parent: Optional[Union[ConfigParamGroup, List[str], str]] = None)\
            -> ConfigParamGroup:
        return self._add_group(name, parent=parent, allow_dynamic_params=self._allow_dynamic_params(), internal=True)

    def add_internal_param(self, name: str, parent: Optional[Union[ConfigParamGroup, List[str], str]] = None,
                           value: Any = None) -> None:
        param = self._add_param(name, parent=parent, internal=True)
        param.value = value

    def read_dict(self, data: Dict) -> None:
        self._read_dict(data)

    def read_file(self, f: str) -> None:
        with open(f, 'r') as stream:
            d = yaml.load(stream, Loader=yaml.FullLoader)
        self.read_dict(d)

    def read_str(self, s: str) -> None:
        d = yaml.load(s, Loader=yaml.FullLoader)
        self.read_dict(d)

    def read_list(self, opts: List[str]) -> None:
        for opt in opts:
            # remove leading --
            opt_strip = opt.strip('-')

            # separate key and value
            eq_idx = opt_strip.rfind('=')
            if eq_idx == -1:
                raise AttributeError("Invalid list parameter '{}'".format(opt))
            key, value = opt_strip[:eq_idx], opt_strip[eq_idx + 1:]
            if len(key) == 0 or len(value) == 0:
                raise AttributeError("Invalid list parameter '{}'".format(opt))

            # get group
            key_list = key.split('.')
            grp: ConfigParamGroup = self
            for subkey in key_list[:-1]:
                if len(subkey) == 0:
                    raise AttributeError("Invalid list parameter '{}'".format(opt))
                if subkey in grp:
                    grp = grp[subkey]
                else:
                    grp = self.add_internal_group(subkey, parent=grp)

            # add param value
            param_key = key_list[-1]
            if len(param_key) == 0:
                raise AttributeError("Invalid list parameter '{}'".format(opt))
            grp[param_key] = value

    def write_file(self, filename: str, invalid: bool = False, internal: bool = False) -> None:
        param_values = self._value_dict(invalid=invalid, internal=internal)
        with open(filename, 'w') as stream:
            yaml.dump(param_values, stream, line_break=True)

    def write_str(self, invalid: bool = False, internal: bool = False) -> str:
        param_values = self._value_dict(invalid=invalid, internal=internal)
        return yaml.dump(param_values, line_break=True)


# alias
Config = ConfigParamGroup


# yaml output representations for OrderedDict and ConfigEnum
def represent_ordereddict(dumper: yaml.Dumper, data: OrderedDict) -> yaml.nodes.Node:
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


def represent_config_enum(dumper: yaml.Dumper, data: ConfigEnum) -> yaml.nodes.Node:
    return dumper.represent_data(data.name)


yaml.add_representer(OrderedDict, represent_ordereddict)
yaml.add_multi_representer(ConfigEnum, represent_config_enum)
