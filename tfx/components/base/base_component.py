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
"""Base class for TFX components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect

from six import with_metaclass

from typing import Any, Dict, Optional, Text

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx import types
from tfx.components.base import base_driver
from tfx.components.base import executor_spec


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class TfxNode(with_metaclass(abc.ABCMeta, object)):
  """Base class for a node in TFX pipeline DAG."""

  def __init__(self, instance_name: Optional[Text] = None):
    # TODO(b/139540680): consider making instance_name private.
    self.instance_name = instance_name
    self._upstream_nodes = set()
    self._downstream_nodes = set()

  @property
  def node_type(self) -> Text:
    return '.'.join([self.__class__.__module__, self.__class__.__name__])

  @property
  @deprecation.deprecated(None,
                          'component_type is deprecated, use node_type instead')
  def component_type(self) -> Text:
    return self.node_type

  # TODO(ruoyu): Consolidate the usage of component identifier. Moving forward,
  # we will have two component level keys:
  # - component_type: the path of the python executor or the image uri of the
  #   executor.
  # - component_id: <component_class_name>.<instance_name>
  @property
  def node_id(self) -> Text:
    """Node id, unique across all TFX nodes in a pipeline.

    If instance name is available, node_id will be:
      <node_class_name>.<instance_name>
    otherwise, node_id will be:
      <node_class_name>

    Returns:
      node id.
    """
    node_class_name = self.__class__.__name__
    if self.instance_name:
      return '{}.{}'.format(node_class_name, self.instance_name)
    else:
      return node_class_name

  @property
  @deprecation.deprecated(None,
                          'component_id is deprecated, use node_id instead')
  def component_id(self) -> Text:
    return self.node_id

  @property
  @abc.abstractmethod
  def inputs(self) -> types.PropertyDictWrapper:
    pass

  @property
  @abc.abstractmethod
  def outputs(self) -> types.PropertyDictWrapper:
    pass

  @property
  def upstream_nodes(self):
    return self._upstream_nodes

  def add_upstream_node(self, upstream_node):
    self._upstream_nodes.add(upstream_node)

  @property
  def downstream_nodes(self):
    return self._downstream_nodes

  def add_downstream_node(self, downstream_node):
    self._downstream_nodes.add(downstream_node)


class BaseComponent(with_metaclass(abc.ABCMeta, TfxNode)):
  """Base class for a TFX pipeline component.

  An instance of a subclass of BaseComponent represents the parameters for a
  single execution of that TFX pipeline component.

  All subclasses of BaseComponent must override the SPEC_CLASS field with the
  ComponentSpec subclass that defines the interface of this component.

  Attributes:
    SPEC_CLASS: a subclass of types.ComponentSpec used by this component
      (required).
    EXECUTOR_SPEC: an instance of executor_spec.ExecutorSpec which describes how
      to execute this component (required).
    DRIVER_CLASS: a subclass of base_driver.BaseDriver as a custom driver for
      this component (optional, defaults to base_driver.BaseDriver).
  """

  # Subclasses must override this property (by specifying a types.ComponentSpec
  # class, e.g. "SPEC_CLASS = MyComponentSpec").
  SPEC_CLASS = _abstract_property()
  # Subclasses must also override the executor spec.
  #
  # Note: EXECUTOR_CLASS has been replaced with EXECUTOR_SPEC. A custom
  # component's existing executor class definition "EXECUTOR_CLASS = MyExecutor"
  # should be replaced with "EXECUTOR_SPEC = ExecutorClassSpec(MyExecutor).
  EXECUTOR_SPEC = _abstract_property()
  # Subclasses will usually use the default driver class, but may override this
  # property as well.
  DRIVER_CLASS = base_driver.BaseDriver

  def __init__(
      self,
      spec: types.ComponentSpec,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      instance_name: Optional[Text] = None):
    """Initialize a component.

    Args:
      spec: types.ComponentSpec object for this component instance.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor specified in the component attribute.
      instance_name: Optional unique identifying name for this instance of the
        component in the pipeline. Required if two instances of the same
        component is used in the pipeline.
    """
    super(BaseComponent, self).__init__(instance_name)
    self.spec = spec
    if custom_executor_spec:
      if not isinstance(custom_executor_spec, executor_spec.ExecutorSpec):
        raise TypeError(
            ('Custom executor spec override %s for %s should be an instance of '
             'ExecutorSpec') % (custom_executor_spec, self.__class__))
    self.executor_spec = (custom_executor_spec or self.__class__.EXECUTOR_SPEC)
    self.driver_class = self.__class__.DRIVER_CLASS
    self._validate_component_class()
    self._validate_spec(spec)

  @classmethod
  def _validate_component_class(cls):
    """Validate that the SPEC_CLASSES property of this class is set properly."""
    if not (inspect.isclass(cls.SPEC_CLASS) and
            issubclass(cls.SPEC_CLASS, types.ComponentSpec)):
      raise TypeError(
          ('Component class %s expects SPEC_CLASS property to be a subclass '
           'of types.ComponentSpec; got %s instead.') % (cls, cls.SPEC_CLASS))
    if not isinstance(cls.EXECUTOR_SPEC, executor_spec.ExecutorSpec):
      raise TypeError((
          'Component class %s expects EXECUTOR_SPEC property to be an instance '
          'of ExecutorSpec; got %s instead.') % (cls, type(cls.EXECUTOR_SPEC)))
    if not (inspect.isclass(cls.DRIVER_CLASS) and
            issubclass(cls.DRIVER_CLASS, base_driver.BaseDriver)):
      raise TypeError(
          ('Component class %s expects DRIVER_CLASS property to be a subclass '
           'of base_driver.BaseDriver; got %s instead.') %
          (cls, cls.DRIVER_CLASS))

  def _validate_spec(self, spec):
    """Verify given spec is valid given the component's SPEC_CLASS."""
    if not isinstance(spec, types.ComponentSpec):
      raise ValueError((
          'BaseComponent (parent class of %s) expects "spec" argument to be an '
          'instance of types.ComponentSpec, got %s instead.') %
                       (self.__class__, spec))
    if not isinstance(spec, self.__class__.SPEC_CLASS):
      raise ValueError(
          ('%s expects the "spec" argument to be an instance of %s; '
           'got %s instead.') %
          (self.__class__, self.__class__.SPEC_CLASS, spec))

  def __repr__(self):
    return ('%s(spec: %s, executor_spec: %s, driver_class: %s, '
            'component_id: %s, inputs: %s, outputs: %s)') % (
                self.__class__.__name__, self.spec, self.executor_spec,
                self.driver_class, self.node_id, self.inputs, self.outputs)

  @property
  def inputs(self) -> types.PropertyDictWrapper:
    return self.spec.inputs

  @property
  def outputs(self) -> types.PropertyDictWrapper:
    return self.spec.outputs

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return self.spec.exec_properties
