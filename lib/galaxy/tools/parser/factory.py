from __future__ import absolute_import

import yaml

from .yaml import YamlToolSource
from .xml import XmlToolSource
from .xml import XmlInputSource
from .interface import InputSource


from galaxy.tools.loader import load_tool as load_tool_xml


import logging
log = logging.getLogger(__name__)


def get_tool_source(config_file, enable_beta_formats=True):
    if not enable_beta_formats:
        tree = load_tool_xml(config_file)
        root = tree.getroot()
        return XmlToolSource(root)

    if config_file.endswith(".yml"):
        log.info("Loading tool from YAML - this is experimental - tool will not function in future.")
        with open(config_file, "r") as f:
            as_dict = yaml.load(f)
            return YamlToolSource(as_dict)
    else:
        tree = load_tool_xml(config_file)
        root = tree.getroot()
        return XmlToolSource(root)


def get_tool_source_from_representation(tool_format, tool_representation):
    log.info("Loading dynamic tool - this is experimental - tool may not function in future.")
    if tool_format == "GalaxyTool":
        return YamlToolSource(tool_representation)
    else:
        raise Exception("Unknown tool representation format [%s]." % tool_format)


def get_input_source(content):
    """ Wraps XML elements in a XmlInputSource until everything
    is consumed using the tool source interface.
    """
    if not isinstance(content, InputSource):
        content = XmlInputSource(content)
    return content
