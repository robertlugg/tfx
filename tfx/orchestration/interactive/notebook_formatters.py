# Copyright 2019 Google LLC. All Rights Reserved.
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
"""TFX IPython notebook formatter integration.

Note: these APIs are **experimental** and major changes to interface and
functionality are expected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

from six.moves import builtins
from typing import Callable, List, Optional, Text, Tuple, Type, Union

from tfx.components.base.base_component import BaseComponent
from tfx.orchestration.interactive.execution_result import ExecutionResult
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel
from tfx.types.component_spec import _PropertyDictWrapper

STATIC_HTML_CONTENTS = u"""<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '\u25bc';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '\u25b6';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
"""


class NotebookFormatter(object):
  """Formats a TFX component in the context of an interactive notebook."""

  _DEFAULT_TITLE_FORMAT = ('<span class="class-name">%s</span>',
                           ['__class__.__name__'])

  def __init__(self,
               cls: Type[object],
               attributes: List[Text] = None,
               title_format: Tuple[Text, List[Union[Text, Callable]]] = None):  # pylint: disable=g-bare-generic
    """Constructs a NotebookFormatter.

    Args:
      cls: The TFX class to be formated by this NotebookFormatter instance.
      attributes: A list of string attributes that are to be displayed by this
        formatter. Can be a nested field specifier with nested attribute names
        separated by "." (e.g. to get `obj.a.b`, specify the attribute string
        "a.b").
      title_format: A 2-tuple consisting of (1) a format string and (2) a list
        of either string attribute names (possible of nested field specifiers as
        in "attributes" above) or callback callable objects taking as input the
        object to be formatted and returning the value for that position of the
        format string. If not specified, the default title format will be used.
    """
    self.cls = cls
    self.attributes = attributes or []
    self.title_format = title_format or NotebookFormatter._DEFAULT_TITLE_FORMAT

  def _extended_getattr(self, obj: object, property_name: Text) -> object:
    """Get a possibly nested attribute of a given object."""
    if callable(property_name):
      return property_name(obj)
    parts = property_name.split('.')
    current = obj
    for part in parts:
      current = getattr(current, part)
    return current

  def render(self,
             obj: object,
             expanded: bool = True,
             seen_elements: Optional[set] = None) -> Text:  # pylint: disable=g-bare-generic
    """Render a given object as an HTML string.

    Args:
      obj: The object to be rendered.
      expanded: Whether the object is to be expanded by default.
      seen_elements: Optionally, a set of seen elements to not re-render to
        prevent a rendering cycle.

    Returns:
      Formatted HTML string representing the object, for notebook display.
    """
    seen_elements = seen_elements or set()
    if id(obj) in seen_elements:
      return '(recursion in rendering object)'
    seen_elements.add(id(obj))
    if not isinstance(obj, self.cls):
      raise ValueError('Expected object of type %s but got %s.' %
                       (self.cls, obj))
    seen_elements.remove(id(obj))
    return STATIC_HTML_CONTENTS + (
        '<div class="tfx-object%s">'
        '<div class = "title" onclick="toggleTfxObject(this)">'
        '<span class="expansion-marker"></span>'
        '%s<span class="deemphasize"> at 0x%x</span></div>%s'
        '</div>') % (' expanded' if expanded else ' collapsed',
                     self.render_title(obj), id(obj),
                     self.render_attributes(obj, seen_elements))

  def render_title(self, obj: object) -> Text:
    """Render the title section of an object."""
    title_format = self.title_format
    values = []
    for property_name in title_format[1]:
      values.append(self._extended_getattr(obj, property_name))
    return title_format[0] % tuple(values)

  def render_value(self,
                   value: object,
                   seen_elements: set) -> object:  # pylint: disable=g-bare-generic
    """Render the value section of an object."""
    if isinstance(value, _PropertyDictWrapper):
      value = value.get_all()
    if isinstance(value, dict):
      value = self.render_dict(value, seen_elements)
    if isinstance(value, list):
      value = self.render_list(value, seen_elements)
    for cls in value.__class__.mro():
      if cls in FORMATTER_REGISTRY:
        value = FORMATTER_REGISTRY[cls].render(
            value, expanded=False, seen_elements=seen_elements)
        break
    return value

  def render_attributes(self,
                        obj: object,
                        seen_elements: set) -> Text:  # pylint: disable=g-bare-generic
    """Render the attributes section of an object."""
    attr_trs = []
    for property_name in self.attributes:
      value = self._extended_getattr(obj, property_name)
      value = self.render_value(value, seen_elements)
      attr_trs.append(
          ('<tr><td class="attr-name">.%s</td>'
           '<td class = "attrvalue">%s</td></tr>') % (property_name, value))
    return '<table class="attr-table">%s</table>' % ''.join(attr_trs)

  def render_dict(self,
                  obj: dict,  # pylint: disable=g-bare-generic
                  seen_elements: set) -> Text:  # pylint: disable=g-bare-generic
    """Render a dictionary table."""
    if not obj:
      return '{}'
    attr_trs = []
    for key, value in obj.items():
      value = self.render_value(value, seen_elements)
      attr_trs.append(('<tr><td class="attr-name">[%r]</td>'
                       '<td class = "attrvalue">%s</td></tr>') % (key, value))
    return '<table class="attr-table">%s</table>' % ''.join(attr_trs)

  def render_list(self,
                  obj: list,  # pylint: disable=g-bare-generic
                  seen_elements: set) -> Text:  # pylint: disable=g-bare-generic
    """Render a list table."""
    if not obj:
      return '[]'
    attr_trs = []
    for i, value in enumerate(obj):
      value = self.render_value(value, seen_elements)
      attr_trs.append(('<tr><td class="attr-name">[%d]</td>'
                       '<td class = "attrvalue">%s</td></tr>') % (i, value))
    return '<table class="attr-table">%s</table>' % ''.join(attr_trs)


def _create_formatters(formatters_spec):
  result = {}
  for cls, kwargs in formatters_spec.items():
    formatter = NotebookFormatter(cls, **kwargs)
    result[cls] = formatter
  return result


FORMATTER_REGISTRY = _create_formatters({
    Artifact: {
        'attributes': ['type_name', 'uri', 'span', 'split'],
        'title_format': (
            ('<span class="class-name">Artifact</span> of type '
             '<span class="class-name">%r</span> (uri: %s)'),
            ['type_name', 'uri']),
    },
    BaseComponent: {
        'attributes': ['inputs', 'outputs', 'exec_properties']
    },
    Channel: {
        'attributes': ['type_name', '_artifacts'],
        'title_format': (
            ('<span class="class-name">Channel</span> of type '
             '<span class="class-name">%r</span> (%d artifact%s)'),
            [
                'type_name', lambda o: len(o._artifacts),  # pylint: disable=protected-access
                lambda o: '' if len(o._artifacts) == 1 else 's'  # pylint: disable=protected-access
            ]),
    },
    ExecutionResult: {
        'attributes': [
            'execution_id', 'component', 'component.inputs', 'component.outputs'
        ]
    },
})


def register_formatters():
  """Register HTML notebook formatters for TFX classes.

  This method registers HTML formatters for TFX classes for display in
  IPython / Jupyter / Colab notebooks. No action will be performed if called
  outside a notebook environment.
  """
  if getattr(builtins, '__IPYTHON__', None):
    # Skip registration if (1) IPython is not installed or (2) if IPython is
    # installed but we are not running in the notebook context (in this case,
    # get_ipython() returns None).
    try:
      ipython = __import__('IPython.core.getipython').get_ipython()
      if not ipython:
        return
    except ImportError:
      return
    html_formatter = ipython.display_formatter.formatters['text/html']
    for cls, formatter in FORMATTER_REGISTRY.items():
      html_formatter.for_type(cls, formatter.render)
