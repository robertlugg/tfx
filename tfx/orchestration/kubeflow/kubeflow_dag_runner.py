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
"""TFX runner for Kubeflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import io
import json
import os
import tarfile
from kfp import compiler
from kfp import dsl
from kfp import gcp
from kfp.compiler._k8s_helper import K8sHelper
from kfp.dsl._metadata import PipelineMeta

from tfx import version
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.kubeflow import base_component
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from typing import Callable, List, Optional, Text
import yaml

# OpFunc represents the type of a function that takes as input a
# dsl.ContainerOp and returns the same object. Common operations such as adding
# k8s secrets, mounting volumes, specifying the use of TPUs and so on can be
# specified as an OpFunc.
# See example usage here:
# https://github.com/kubeflow/pipelines/blob/master/sdk/python/kfp/gcp.py
OpFunc = Callable[[dsl.ContainerOp], dsl.ContainerOp]

# Default secret name for GCP credentials. This secret is installed as part of
# a typical Kubeflow installation when the platform is GKE.
_KUBEFLOW_GCP_SECRET_NAME = 'user-gcp-sa'

# Default TFX container image to use in Kubeflow. Overrideable by 'tfx_image'
# pipeline property.
_KUBEFLOW_TFX_IMAGE = 'tensorflow/tfx:%s' % (version.__version__)


def get_default_pipeline_operator_funcs() -> List[OpFunc]:
  """Returns a default list of pipeline operator functions.

  Returns:
    A list of functions with type OpFunc.
  """
  # Enables authentication for GCP services in a typical GKE Kubeflow
  # installation.
  gcp_secret_op = gcp.use_gcp_secret(_KUBEFLOW_GCP_SECRET_NAME)

  return [gcp_secret_op]


def get_default_kubeflow_metadata_config(
) -> kubeflow_pb2.KubeflowMetadataConfig:
  """Returns the default metadata connection config for Kubeflow.

  Returns:
    A config proto that will be serialized as JSON and passed to the running
    container so the TFX component driver is able to communicate with MLMD in
    a Kubeflow cluster.
  """
  # The default metadata configuration for a Kubeflow cluster can be found
  # here:
  # https://github.com/kubeflow/manifests/blob/master/metadata/base/metadata-db-deployment.yaml

  # If deploying Kubeflow Pipelines outside of Kubeflow, that configuration
  # lives here:
  # https://github.com/kubeflow/pipelines/blob/master/manifests/kustomize/base/mysql/mysql-deployment.yaml

  config = kubeflow_pb2.KubeflowMetadataConfig()
  # The environment variable to use to obtain the MySQL service host in the
  # cluster that is backing Kubeflow Metadata.
  config.mysql_db_service_host.environment_variable = 'METADATA_DB_SERVICE_HOST'
  # The environment variable to use to obtain the MySQL service port in the
  # cluster that is backing Kubeflow Metadata.
  config.mysql_db_service_port.environment_variable = 'METADATA_DB_SERVICE_PORT'
  # The MySQL database name to use.
  config.mysql_db_name.value = 'metadb'
  # The MySQL database username.
  config.mysql_db_user.value = 'root'
  # The MySQL database password. It is currently set to `test` for the
  # default install of Kubeflow Metadata:
  # https://github.com/kubeflow/manifests/blob/master/metadata/base/metadata-db-secret.yaml
  # Note that you should ideally use k8s secrets for username/passwords. If you
  # do so, you can change this setting so the container obtains the value at
  # runtime from the secred mounted as an environment variable.
  config.mysql_db_password.value = 'test'

  return config


class KubeflowDagRunnerConfig(object):
  """Runtime configuration parameters specific to execution on Kubeflow."""

  def __init__(
      self,
      pipeline_operator_funcs: Optional[List[OpFunc]] = None,
      tfx_image: Optional[Text] = None,
      kubeflow_metadata_config: Optional[
          kubeflow_pb2.KubeflowMetadataConfig] = None,
  ):
    """Creates a KubeflowDagRunnerConfig object.

    The user can use pipeline_operator_funcs to apply modifications to
    ContainerOps used in the pipeline. For example, to ensure the pipeline
    steps mount a GCP secret, and a Persistent Volume, one can create config
    object like so:

      from kfp import gcp, onprem
      mount_secret_op = gcp.use_secret('my-secret-name)
      mount_volume_op = onprem.mount_pvc(
        "my-persistent-volume-claim",
        "my-volume-name",
        "/mnt/volume-mount-path")

      config = KubeflowDagRunnerConfig(
        pipeline_operator_funcs=[mount_secret_op, mount_volume_op]
      )

    Args:
      pipeline_operator_funcs: A list of ContainerOp modifying functions that
        will be applied to every container step in the pipeline.
      tfx_image: The TFX container image to use in the pipeline.
      kubeflow_metadata_config: Runtime configuration to use to connect to
        Kubeflow metadata.
    """
    self.pipeline_operator_funcs = (
        pipeline_operator_funcs or get_default_pipeline_operator_funcs())
    self.tfx_image = tfx_image or _KUBEFLOW_TFX_IMAGE
    self.kubeflow_metadata_config = (
        kubeflow_metadata_config or get_default_kubeflow_metadata_config())


class KubeflowDagRunner(tfx_runner.TfxRunner):
  """Kubeflow Pipelines runner.

  Constructs a pipeline definition YAML file based on the TFX logical pipeline.
  """

  def __init__(self,
               output_dir: Optional[Text] = None,
               config: Optional[KubeflowDagRunnerConfig] = None):
    """Initializes KubeflowDagRunner for compiling a Kubeflow Pipeline.

    Args:
      output_dir: An optional output directory into which to output the pipeline
        definition files. Defaults to the current working directory.
      config: An optional KubeflowDagRunnerConfig object to specify runtime
        configuration when running the pipeline under Kubeflow.
    """
    self._output_dir = output_dir or os.getcwd()
    self._config = config or KubeflowDagRunnerConfig()
    self._compiler = compiler.Compiler()

  def _construct_pipeline_graph(self, pipeline: tfx_pipeline.Pipeline):
    """Constructs a Kubeflow Pipeline graph.

    Args:
      pipeline: The logical TFX pipeline to base the construction on.
    """
    component_to_kfp_op = {}

    # Assumption: There is a partial ordering of components in the list, i.e.,
    # if component A depends on component B and C, then A appears after B and C
    # in the list.
    for component in pipeline.components:
      # Keep track of the set of upstream dsl.ContainerOps for this component.
      depends_on = set()

      for upstream_component in component.upstream_nodes:
        depends_on.add(component_to_kfp_op[upstream_component])

      kfp_component = base_component.BaseComponent(
          component=component,
          depends_on=depends_on,
          pipeline=pipeline,
          tfx_image=self._config.tfx_image,
          kubeflow_metadata_config=self._config.kubeflow_metadata_config)

      for operator in self._config.pipeline_operator_funcs:
        kfp_component.container_op.apply(operator)

      component_to_kfp_op[component] = kfp_component.container_op

  def _pipeline_injection(self, pipeline: dsl.Pipeline):
    """Inject pipeline level info and sanitize op/param names."""
    # TODO(b/139957573): Currently duplicate logic from kfp.compiler. This
    # should be simplified after kfp has a better modularized compiling method.

    # Sanitize operator names and param names
    sanitized_ops = {}
    # pipeline level artifact location
    artifact_location = pipeline.conf.artifact_location

    for op in pipeline.ops.values():
      # inject pipeline level artifact location into if the op does not have
      # an artifact location config already.
      if hasattr(op, 'artifact_location'):
        if artifact_location and not op.artifact_location:
          op.artifact_location = artifact_location

      sanitized_name = K8sHelper.sanitize_k8s_name(op.name)
      op.name = sanitized_name
      for param in op.outputs.values():
        param.name = K8sHelper.sanitize_k8s_name(param.name)
        if param.op_name:
          param.op_name = K8sHelper.sanitize_k8s_name(param.op_name)
      if op.output is not None:
        op.output.name = K8sHelper.sanitize_k8s_name(op.output.name)
        op.output.op_name = K8sHelper.sanitize_k8s_name(op.output.op_name)
      if op.dependent_names:
        op.dependent_names = \
          [K8sHelper.sanitize_k8s_name(name) for name in op.dependent_names]
      if isinstance(op, dsl.ContainerOp) and op.file_outputs is not None:
        sanitized_file_outputs = {}
        for key in op.file_outputs.keys():
          sanitized_file_outputs[K8sHelper.sanitize_k8s_name(key)] \
            = op.file_outputs[key]
        op.file_outputs = sanitized_file_outputs
      elif isinstance(op, dsl.ResourceOp) and op.attribute_outputs is not None:
        sanitized_attribute_outputs = {}
        for key in op.attribute_outputs.keys():
          sanitized_attribute_outputs[K8sHelper.sanitize_k8s_name(key)] = \
            op.attribute_outputs[key]
        op.attribute_outputs = sanitized_attribute_outputs
      sanitized_ops[sanitized_name] = op
    pipeline.ops = sanitized_ops

    return self._compiler._create_pipeline_workflow(  # pylint: disable=protected-access
        [], pipeline, pipeline.conf.op_transformers)

  def run(self, pipeline: tfx_pipeline.Pipeline):
    """Compiles and outputs a Kubeflow Pipeline YAML definition file.

    Args:
      pipeline: The logical TFX pipeline to use when building the Kubeflow
        pipeline.
    """

    # TODO(b/128836890): Add pipeline parameters after removing the usage of
    # pipeline function.
    sanitized_name = K8sHelper.sanitize_k8s_name(
        pipeline.pipeline_args['pipeline_name'])
    with dsl.Pipeline(sanitized_name) as dsl_pipeline:
      # Constructs a Kubeflow pipeline.
      # Creates Kubeflow ContainerOps for each TFX component encountered in the
      # logical pipeline definition and append them to the pipeline object.
      self._construct_pipeline_graph(pipeline)

    workflow = self._pipeline_injection(dsl_pipeline)
    pipeline_meta = PipelineMeta(
        name=sanitized_name, description='')

    workflow.setdefault('metadata', {}).setdefault(
        'annotations', {})['pipelines.kubeflow.org/pipeline_spec'] = json.dumps(
            pipeline_meta.to_dict(), sort_keys=True)
    yaml_text = yaml.dump(workflow, default_flow_style=False)

    pipeline_name = pipeline.pipeline_args['pipeline_name']
    # TODO(b/134680219): Allow users to specify the extension. Specifying
    # .yaml will compile the pipeline directly into a YAML file. Kubeflow
    # backend recognizes .tar.gz, .zip, and .yaml today.
    pipeline_file = os.path.join(self._output_dir, pipeline_name + '.tar.gz')
    with tarfile.open(pipeline_file, 'w:gz') as tar:
      with contextlib.closing(io.BytesIO(yaml_text.encode())) as yaml_file:
        tarinfo = tarfile.TarInfo('pipeline.yaml')
        tarinfo.size = len(yaml_file.getvalue())
        tar.addfile(tarinfo, fileobj=yaml_file)
