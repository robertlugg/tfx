# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.tools.cli.handler.kubeflow_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os
import sys
import tarfile
import tempfile
import mock
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import kubeflow_handler
from tfx.utils import io_utils


def _MockSubprocess(cmd, env):  # pylint: disable=invalid-name, unused-argument
  # Store pipeline_args in a pickle file
  pipeline_args_path = env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH]
  pipeline_args = {'pipeline_name': 'chicago_taxi_pipeline_kubeflow'}
  with open(pipeline_args_path, 'w') as f:
    json.dump(pipeline_args, f)

  chicago_taxi_pipeline_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
  pipeline_path = os.path.join(chicago_taxi_pipeline_dir,
                               'test_pipeline_kubeflow_1.py')
  # Store pipeline package
  output_filename = os.path.join(chicago_taxi_pipeline_dir,
                                 'chicago_taxi_pipeline_kubeflow.tar.gz')
  with tarfile.open(output_filename, 'w:gz') as tar:
    tar.add(pipeline_path)
  return 0


class _MockUploadResponse(object):
  """Mock upload response object."""

  def __init__(self, config):
    self.host = config['host']
    self.client_id = config['client_id']
    self.namespace = config['namespace']
    self.id = config['id']
    self.name = config['name']


class _MockClientClass(object):

  def __init__(self, host, client_id, namespace):

    self.config = {
        'host': host,
        'client_id': client_id,
        'namespace': namespace,
        'id': 'fake_pipeline_id',
        'name': 'fake_pipeline_name'
    }  # pylint: disable=invalid-name, unused-variable
    self._output_dir = os.path.join(tempfile.gettempdir(), 'output_dir')
    self._pipelines_api = _MockPipelineApi()
    self._experiment_api = _MockExperimentApi()
    self._run_api = _MockRunApi()

  def upload_pipeline(self, pipeline_package_path, pipeline_name):  # pylint: disable=invalid-name, unused-argument
    io_utils.copy_file(
        pipeline_package_path,
        os.path.join(self._output_dir, os.path.basename(pipeline_package_path)),
        overwrite=True)
    return _MockUploadResponse(self.config)

  def create_experiment(self, name):
    return self._experiment_api.create_experiment(name)

  def get_experiment(self, experiment_id=None, experiment_name=None):  # pylint: disable=unused-argument
    return self._experiment_api.get_experiment(experiment_id)

  def run_pipeline(self, experiment_id, job_name, pipeline_id=None):  # pylint: disable=unused-argument
    return self._pipelines_api.run_pipeline()

  def list_pipelines(self):
    return self._pipelines_api.list_pipelines()

  def list_runs(self, experiment_id):
    return self._run_api.list_runs(experiment_id)

  def get_run(self, run_id):
    return self._run_api.get_run(run_id)


class _MockPipelineApi(object):

  def delete_pipeline(self, id):  # pylint: disable=redefined-builtin, invalid-name
    pass

  def get_pipeline(self, id):  # pylint: disable=redefined-builtin, invalid-name
    return id

  def list_pipelines(self):
    pass

  def run_pipeline(self):
    return _MockRunResponse('run_id', 'Running', datetime.datetime.now())


class _MockExperimentResponse(object):

  def __init__(self, experiment_name, experiment_id):  # pylint: disable=redefined-builtin
    self.name = experiment_name
    self.id = experiment_id


class _MockExperimentApi(object):

  def create_experiment(self, name):
    return _MockExperimentResponse(name, 'fake_id')

  def get_experiment(self, id):  # pylint: disable=redefined-builtin
    return _MockExperimentResponse('fake_name', id)

  def delete_experiment(self, id):  # pylint: disable=redefined-builtin, invalid-name
    pass


class _MockRunResponse(object):

  def __init__(self, run_id, status, created_at):
    self.id = run_id
    self.status = status
    self.created_at = created_at


class _Runs(object):

  def __init__(self, runs):
    self.runs = runs


class _MockRunApi(object):

  def delete_run(self, run_id):
    pass

  def terminate_run(self, run_id):
    pass

  def get_run(self, run_id):
    return _MockRunResponse(run_id, 'Running', datetime.datetime.now())

  def list_runs(self, experiment_id):  # pylint: disable=unused-argument
    run_1 = _MockRunResponse('1', 'Success', datetime.datetime.now())
    run_2 = _MockRunResponse('2', 'Failed', datetime.datetime.now())
    return _Runs([run_1, run_2])


class KubeflowHandlerTest(tf.test.TestCase):

  def setUp(self):
    super(KubeflowHandlerTest, self).setUp()
    self._home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._original_home_value = os.environ.get('HOME', '')
    os.environ['HOME'] = self._home
    self._original_kubeflow_home_value = os.environ.get('KUBEFLOW_HOME', '')
    os.environ['KUBEFLOW_HOME'] = os.path.join(os.environ['HOME'], 'kubeflow')

    # Flags for handler.
    self.engine = 'kubeflow'
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_kubeflow_1.py')
    self.pipeline_name = 'chicago_taxi_pipeline_kubeflow'
    self.pipeline_package_path = os.path.join(
        os.getcwd(), 'chicago_taxi_pipeline_kubeflow.tar.gz')

    # Kubeflow client params.
    self.endpoint = 'dummyEndpoint'
    self.namespace = 'kubeflow'
    self.iap_client_id = 'dummyID'

    # Pipeline args for mocking subprocess.
    self.pipeline_args = {'pipeline_name': 'chicago_taxi_pipeline_kubeflow'}

  def tearDown(self):
    super(KubeflowHandlerTest, self).tearDown()
    os.environ['HOME'] = self._original_home_value
    os.environ['KUBEFLOW_HOME'] = self._original_kubeflow_home_value

  # TODO(b/140954873): Change the following test after Kubeflow e2e test.
  @mock.patch('kfp.Client', _MockClientClass)
  def testCheckPipelinePackagePathDefaultPath(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: None
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    pipeline_args = handler._extract_pipeline_args()
    handler._check_pipeline_package_path(pipeline_args[labels.PIPELINE_NAME])
    self.assertEqual(
        handler.flags_dict[labels.PIPELINE_PACKAGE_PATH],
        os.path.join(os.getcwd(),
                     '{}.tar.gz'.format(pipeline_args[labels.PIPELINE_NAME])))

  @mock.patch('kfp.Client', _MockClientClass)
  def testCheckPipelinePackagePathWrongPath(self):
    flags_dict = {
        labels.ENGINE_FLAG:
            self.engine,
        labels.PIPELINE_DSL_PATH:
            self.pipeline_path,
        labels.ENDPOINT:
            self.endpoint,
        labels.IAP_CLIENT_ID:
            self.iap_client_id,
        labels.NAMESPACE:
            self.namespace,
        labels.PIPELINE_PACKAGE_PATH:
            os.path.join(self.chicago_taxi_pipeline_dir,
                         '{}.tar.gz'.format(self.pipeline_name))
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    pipeline_args = handler._extract_pipeline_args()
    with self.assertRaises(SystemExit) as err:
      handler._check_pipeline_package_path(pipeline_args[labels.PIPELINE_NAME])
    self.assertEqual(
        str(err.exception),
        'Pipeline package not found at {}. When --package_path is unset, it will try to find the workflow file, "<pipeline_name>.tar.gz" in the current directory.'
        .format(flags_dict[labels.PIPELINE_PACKAGE_PATH]))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testSavePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler._save_pipeline(self.pipeline_args)
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertTrue(os.path.join(handler_pipeline_path, 'pipeline_args.json'))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testCreatePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))
    handler.create_pipeline()
    self.assertTrue(tf.io.gfile.exists(handler_pipeline_path))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testCreatePipelineExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.create_pipeline()
    # Run create_pipeline again to test.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" already exists.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testUpdatePipeline(self):
    # First create pipeline with test_pipeline.py
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_kubeflow_1.py')
    flags_dict_1 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_1,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict_1)
    handler.create_pipeline()

    # Update test_pipeline and run update_pipeline
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_kubeflow_2.py')
    flags_dict_2 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_2,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict_2)
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertTrue(tf.io.gfile.exists(handler_pipeline_path))
    handler.update_pipeline()
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testUpdatePipelineNoPipeline(self):
    # Update pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testCompilePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.compile_pipeline()
    self.assertIn('Pipeline compiled successfully', captured.contents())
    self.assertIn('Pipeline package path', captured.contents())

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testDeletePipeline(self):
    # Create pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.create_pipeline()

    # Delete pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.delete_pipeline()
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testDeletePipelineNonExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testCreateRun(self):
    # Create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.create_pipeline()

    # Run pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.create_run()
    self.assertIn('Run created for pipeline: ', captured.contents())

  @mock.patch('kfp.Client', _MockClientClass)
  def testCreateRunNoPipeline(self):
    # Run pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.create_run()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def testListRuns(self):
    # Create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.create_pipeline()

    # List pipelines.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_runs()
    self.assertIn('pipeline_name', captured.contents())

  @mock.patch('kfp.Client', _MockClientClass)
  def testListRunsNoPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.list_runs()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))


if __name__ == '__main__':
  tf.test.main()
