import contextlib
import json
import os.path
import time
import unittest
from functools import wraps
from operator import itemgetter

import cwltest
import requests
import yaml

from pkg_resources import resource_string
from six import StringIO

from galaxy.tools.verify.test_data import TestDataResolver
from galaxy.tools.cwl.util import (
    FileUploadTarget,
    DirectoryUploadTarget,
    download_output,
    galactic_job_json,
    invocation_to_output,
    output_to_cwl_json,
    tool_response_to_output,
)
from galaxy.util import galaxy_root_path

from . import api_asserts
from .workflows_format_2 import (
    convert_and_import_workflow,
    ImporterGalaxyInterface,
)


CWL_TOOL_DIRECTORY = os.path.join(galaxy_root_path, "test", "functional", "tools", "cwl_tools")

# Simple workflow that takes an input and call cat wrapper on it.
workflow_str = resource_string(__name__, "data/test_workflow_1.ga")
# Simple workflow that takes an input and filters with random lines twice in a
# row - first grabbing 8 lines at random and then 6.
workflow_random_x2_str = resource_string(__name__, "data/test_workflow_2.ga")

UPLOAD_VIA = "path"  # or content, but content breaks down for empty uploads, tar, etc...

DEFAULT_TIMEOUT = 60  # Secs to wait for state to turn ok

SKIP_FLAKEY_TESTS_ON_ERROR = os.environ.get("GALAXY_TEST_SKIP_FLAKEY_TESTS_ON_ERROR", None)


def flakey(method):

    @wraps(method)
    def wrapped_method(test_case, *args, **kwargs):
        try:
            method(test_case, *args, **kwargs)
        except unittest.SkipTest:
            raise
        except Exception:
            if SKIP_FLAKEY_TESTS_ON_ERROR:
                raise unittest.SkipTest("Error encountered during test marked as @flakey.")
            else:
                raise

    return wrapped_method


def skip_without_tool(tool_id):
    """Decorate an API test method as requiring a specific tool.

    Have test framework skip the test case if the tool is unavailable.
    """

    def method_wrapper(method):

        def get_tool_ids(api_test_case):
            index = api_test_case.galaxy_interactor.get("tools", data=dict(in_panel=False))
            tools = index.json()
            # In panels by default, so flatten out sections...
            tool_ids = [itemgetter("id")(_) for _ in tools]
            return tool_ids

        @wraps(method)
        def wrapped_method(api_test_case, *args, **kwargs):
            _raise_skip_if(tool_id not in get_tool_ids(api_test_case))
            return method(api_test_case, *args, **kwargs)

        return wrapped_method

    return method_wrapper


def skip_without_datatype(extension):
    """Decorate an API test method as requiring a specific datatype.

    Have test framework skip the test case if the datatype is unavailable.
    """

    def has_datatype(api_test_case):
        index_response = api_test_case.galaxy_interactor.get("datatypes")
        assert index_response.status_code == 200, "Failed to fetch datatypes for target Galaxy."
        datatypes = index_response.json()
        assert isinstance(datatypes, list)
        return extension in datatypes

    def method_wrapper(method):
        @wraps(method)
        def wrapped_method(api_test_case, *args, **kwargs):
            _raise_skip_if(not has_datatype(api_test_case))
            method(api_test_case, *args, **kwargs)

        return wrapped_method

    return method_wrapper


def summarize_instance_history_on_error(method):
    @wraps(method)
    def wrapped_method(api_test_case, *args, **kwds):
        try:
            method(api_test_case, *args, **kwds)
        except Exception:
            api_test_case.dataset_populator._summarize_history(api_test_case.history_id)
            raise

    return wrapped_method


def _raise_skip_if(check):
    if check:
        from nose.plugins.skip import SkipTest
        raise SkipTest()


# Deprecated mixin, use dataset populator instead.
# TODO: Rework existing tests to target DatasetPopulator in a setup method instead.
class TestsDatasets:

    def _new_dataset(self, history_id, content='TestData123', **kwds):
        return DatasetPopulator(self.galaxy_interactor).new_dataset(history_id, content=content, **kwds)

    def _wait_for_history(self, history_id, assert_ok=False):
        return DatasetPopulator(self.galaxy_interactor).wait_for_history(history_id, assert_ok=assert_ok)

    def _new_history(self, **kwds):
        return DatasetPopulator(self.galaxy_interactor).new_history(**kwds)

    def _upload_payload(self, history_id, content, **kwds):
        return DatasetPopulator(self.galaxy_interactor).upload_payload(history_id, content, **kwds)

    def _run_tool_payload(self, tool_id, inputs, history_id, **kwds):
        return DatasetPopulator(self.galaxy_interactor).run_tool_payload(tool_id, inputs, history_id, **kwds)


class CwlRun(object):

    def __init__(self, dataset_populator, history_id):
        self.dataset_populator = dataset_populator
        self.history_id = history_id

    def get_output_as_object(self, output_name, download_folder=None):
        galaxy_output = self._output_name_to_object(output_name)

        def get_metadata(history_content_type, content_id):
            if history_content_type == "dataset":
                return self.dataset_populator.get_history_dataset_details(self.history_id, dataset_id=content_id)
            else:
                # Don't wait - we've already done that, history might be "new"
                return self.dataset_populator.get_history_collection_details(self.history_id, content_id=content_id, wait=False)

        def get_dataset(dataset_details, filename=None):
            content = self.dataset_populator.get_history_dataset_content(self.history_id, dataset_id=dataset_details["id"], filename=filename)
            if filename is None:
                basename = dataset_details.get("cwl_file_name")
                if not basename:
                    basename = dataset_details.get("name")
            else:
                basename = os.path.basename(filename)
            return {"content": content, "basename": basename}

        def get_extra_files(dataset_details):
            return self.dataset_populator.get_history_dataset_extra_files(self.history_id, dataset_id=dataset_details["id"])
        output = output_to_cwl_json(
            galaxy_output,
            get_metadata,
            get_dataset,
            get_extra_files,
            pseduo_location=True,
        )
        if download_folder:
            download_path = os.path.join(download_folder, output["basename"])
            download_output(galaxy_output, get_metadata, get_dataset, get_extra_files, download_path)
            output["path"] = download_path
            output["location"] = "file://%s" % download_path 
        return output
        

class CwlToolRun(CwlRun):

    def __init__(self, dataset_populator, history_id, run_response):
        self.dataset_populator = dataset_populator
        self.history_id = history_id
        self.run_response = run_response

    @property
    def job_id(self):
        return self.run_response["jobs"][0]["id"]

    def output(self, output_index):
        return self.run_response["outputs"][output_index]

    def output_collection(self, output_index):
        return self.run_response["output_collections"][output_index]

    def _output_name_to_object(self, output_name):
        return tool_response_to_output(self.run_response, self.history_id, output_name)

    def wait(self):
        final_state = self.dataset_populator.wait_for_job(self.job_id)
        assert final_state == "ok"


class CwlWorkflowRun(CwlRun):

    def __init__(self, dataset_populator, workflow_populator, history_id, workflow_id, invocation_id):
        self.dataset_populator = dataset_populator
        self.workflow_populator = workflow_populator
        self.history_id = history_id
        self.workflow_id = workflow_id
        self.invocation_id = invocation_id

    def _output_name_to_object(self, output_name):
        invocation_response = self.dataset_populator._get("workflows/%s/invocations/%s" % (self.invocation_id, self.workflow_id))
        api_asserts.assert_status_code_is(invocation_response, 200)
        invocation = invocation_response.json()
        return invocation_to_output(invocation, self.history_id, output_name)

    def wait(self):
        self.workflow_populator.wait_for_invocation_and_jobs(
            self.history_id, self.workflow_id, self.invocation_id
        )


class CwlPopulator(object):

    def __init__(self, dataset_populator, workflow_populator):
        self.dataset_populator = dataset_populator
        self.workflow_populator = workflow_populator

    def run_cwl_artifact(
        self, tool_id, json_path=None, job=None, test_data_directory=None, history_id=None, assert_ok=True, tool_or_workflow="tool", upload_via=UPLOAD_VIA
    ):
        if test_data_directory is None and json_path is not None:
            test_data_directory = os.path.dirname(json_path)
        if json_path is not None:
            assert job is None
            with open(json_path, "r") as f:
                if json_path.endswith(".yml") or json_path.endswith(".yaml"):
                    job_as_dict = yaml.load(f)
                else:
                    job_as_dict = json.load(f)
        else:
            job_as_dict = job
        if history_id is None:
            history_id = self.dataset_populator.new_history()

        def upload_func(upload_target):
            if isinstance(upload_target, FileUploadTarget):
                path = upload_target.path

                if upload_via == "path":
                    content = "file://%s" % path
                else:
                    with open(path, "rb") as f:
                        content = f.read()

                name = os.path.basename(path)

                extra_inputs = dict()
                if upload_target.secondary_files:
                    assert upload_via == "path"
                    extra_inputs["files_1|url_paste"] = "file://%s" % upload_target.secondary_files
                    extra_inputs["files_1|type"] = "upload_dataset"
                    extra_inputs["files_1|auto_decompress"] = True
                    extra_inputs["file_count"] = "2"
                    extra_inputs["force_composite"] = "True"

                return self.dataset_populator.new_dataset_request(
                    history_id=history_id,
                    content=content,
                    file_type="auto",
                    name=name,
                    auto_decompress=False,
                    extra_inputs=extra_inputs,
                ).json()
            elif isinstance(upload_target, DirectoryUploadTarget):
                path = upload_target.tar_path

                if upload_via == "path":
                    # TODO: basename?
                    payload = self.dataset_populator.upload_payload(
                        history_id, 'file://%s' % path, ext="tar",
                    )
                else:
                    raise NotImplementedError()

                create_response = self.dataset_populator._post("tools", data=payload)
                assert create_response.status_code == 200

                convert_response = self.dataset_populator.run_tool(
                    tool_id="CONVERTER_tar_to_directory",
                    inputs={"input1": {"src": "hda", "id": create_response.json()["outputs"][0]["id"]}},
                    history_id=history_id,
                )
                assert "outputs" in convert_response, convert_response
                return convert_response
            else:
                content = json.dumps(upload_target.object)
                return self.dataset_populator.new_dataset_request(
                    history_id=history_id,
                    content=content,
                    file_type="expression.json",
                ).json()

        def create_collection_func(element_identifiers, collection_type):
            payload = {
                "name": "dataset collection",
                "instance_type": "history",
                "history_id": history_id,
                "element_identifiers": json.dumps(element_identifiers),
                "collection_type": collection_type,
                "fields": None if collection_type != "record" else "auto",
            }
            response = self.dataset_populator._post("dataset_collections", data=payload)
            assert response.status_code == 200
            return response.json()

        job_as_dict, datasets_uploaded = galactic_job_json(
            job_as_dict,
            test_data_directory,
            upload_func,
            create_collection_func,
            tool_or_workflow=tool_or_workflow,
        )
        if datasets_uploaded:
            self.dataset_populator.wait_for_history(history_id=history_id, assert_ok=True)
        if tool_or_workflow == "tool":
            tool_hash = None

            if os.path.exists(tool_id):
                # Assume it is a file not a tool_id.
                with open(tool_id, "r") as f:
                    representation = yaml.load(f)
                    if "id" not in representation:
                        representation["id"] = os.path.splitext(os.path.basename(tool_id))[0]
                        tool_directory = os.path.abspath(os.path.dirname(tool_id))

                    dynamic_tool = self.dataset_populator.create_tool(representation, tool_directory=tool_directory)
                    tool_id = dynamic_tool["tool_id"]
                    tool_hash = dynamic_tool["tool_hash"]
                    assert tool_id, dynamic_tool

            run_response = self.dataset_populator.run_tool(None, job_as_dict, history_id, inputs_representation="cwl", assert_ok=assert_ok, tool_hash=tool_hash)
            run_object = CwlToolRun(self.dataset_populator, history_id, run_response)
            if assert_ok:
                try:
                    final_state = self.dataset_populator.wait_for_job(run_object.job_id)
                    assert final_state == "ok"
                except Exception:
                    self.dataset_populator._summarize_history(history_id)
                    raise

            return run_object
        else:
            route = "workflows"
            path = os.path.join(tool_id)
            data = dict(
                from_path=path,
            )
            upload_response = self.dataset_populator._post(route, data=data)
            api_asserts.assert_status_code_is(upload_response, 200)
            workflow = upload_response.json()
            workflow_id = workflow["id"]

            workflow_request = dict(
                history="hist_id=%s" % history_id,
                workflow_id=workflow_id,
                inputs=json.dumps(job_as_dict),
                inputs_by="name",
            )
            url = "workflows/%s/invocations" % workflow_id
            invocation_response = self.dataset_populator._post(url, data=workflow_request)
            api_asserts.assert_status_code_is(invocation_response, 200)
            invocation_id = invocation_response.json()["id"]
            return CwlWorkflowRun(self.dataset_populator, self.workflow_populator, history_id, workflow_id, invocation_id)

    def get_conformance_test(self, version, doc):
        conformance_tests = yaml.load(open(os.path.join(CWL_TOOL_DIRECTORY, str(version), "conformance_tests.yaml"), "r"))
        for test in conformance_tests:
            if test.get("doc") == doc:
                return test
        raise Exception("No such doc found %s" % doc)

    def run_conformance_test(self, version, doc):
        test = self.get_conformance_test(version, doc)
        tool = os.path.join(CWL_TOOL_DIRECTORY, test["tool"])
        job = os.path.join(CWL_TOOL_DIRECTORY, test["job"])
        try:
            run = self.run_cwl_job(tool, job)
        except Exception:
            # Should fail so this is good!
            if test.get("should_fail", False):
                return True
            raise

        expected_outputs = test["output"]
        try:
            for key, value in expected_outputs.items():
                actual_output = run.get_output_as_object(key)
                cwltest.compare(value, actual_output)
        except Exception:
            self.dataset_populator._summarize_history(run.history_id)
            raise

    def run_cwl_job(self, tool, job):
        tool_or_workflow = self.guess_artifact_type(tool)
        run = self.run_workflow_job(tool, job, tool_or_workflow=tool_or_workflow)
        assert run.history_id
        return run

    def guess_artifact_type(self, path):
        # TODO: handle IDs within files and use galaxy-lib functionality for this.
        tool_or_workflow = "workflow"
        try:
            with open(path, "r") as f:
                artifact = yaml.load(f)

            tool_or_workflow = "tool" if artifact["class"] != "Workflow" else "workflow"

        except Exception:
            print("Failed to guess artifact type for [%s] - assuming worklfow" % path)
        return tool_or_workflow

    def run_workflow_job(self, workflow_path, job_path, history_id=None, tool_or_workflow="workflow"):
        if history_id is None:
            history_id = self.dataset_populator.new_history()
        if not os.path.isabs(workflow_path):
            workflow_path = os.path.join(CWL_TOOL_DIRECTORY, workflow_path)
        if not os.path.isabs(job_path):
            job_path = os.path.join(CWL_TOOL_DIRECTORY, job_path)
        run_object = self.run_cwl_artifact(
            workflow_path,
            job_path,
            history_id=history_id,
            tool_or_workflow=tool_or_workflow,
        )
        run_object.wait()
        return run_object


class BaseDatasetPopulator(object):
    """ Abstract description of API operations optimized for testing
    Galaxy - implementations must implement _get, _post and _delete.
    """

    def new_dataset(self, history_id, content=None, wait=False, **kwds):
        run_response = self.new_dataset_request(history_id, content=content, wait=wait, **kwds)
        assert run_response.status_code == 200, "Failed to create new dataset with response: %s" % run_response.content
        return run_response.json()["outputs"][0]

    def new_dataset_request(self, history_id, content=None, wait=False, **kwds):
        if content is None and "ftp_files" not in kwds:
            content = "TestData123"
        payload = self.upload_payload(history_id, content=content, **kwds)
        run_response = self.tools_post(payload)
        if wait:
            self.wait_for_tool_run(history_id, run_response, assert_ok=kwds.get('assert_ok', True))
        return run_response

    def fetch(self, payload, assert_ok=True, timeout=DEFAULT_TIMEOUT):
        tool_response = self._post("tools/fetch", data=payload)
        if assert_ok:
            job = self.check_run(tool_response)
            self.wait_for_job(job["id"], timeout=timeout)

            job = tool_response.json()["jobs"][0]
            details = self.get_job_details(job["id"]).json()
            assert details["state"] == "ok", details

        return tool_response

    def wait_for_tool_run(self, history_id, run_response, timeout=DEFAULT_TIMEOUT, assert_ok=True):
        job = self.check_run(run_response)
        self.wait_for_job(job["id"], timeout=timeout)
        self.wait_for_history(history_id, assert_ok=assert_ok, timeout=timeout)
        return run_response

    def check_run(self, run_response):
        run = run_response.json()
        assert run_response.status_code == 200, run
        job = run["jobs"][0]
        return job

    def wait_for_history(self, history_id, assert_ok=False, timeout=DEFAULT_TIMEOUT, hack_allow_new=False):
        try:
            skip_states = ["running", "queued", "new", "ready"]
            ok_states = ["ok"]
            if hack_allow_new:
                # If there are empty collections in the history for instance with no datasets,
                # the history summary will be new even though there are jobs in the history.
                skip_states = ["running", "queued", "ready"]
                ok_states = ["new", "ok"]

            return wait_on_state(lambda: self._get("histories/%s" % history_id), desc="history state", assert_ok=assert_ok, timeout=timeout, skip_states=skip_states, ok_states=ok_states)
        except AssertionError:
            self._summarize_history(history_id)
            raise

    def wait_for_history_jobs(self, history_id, assert_ok=False, timeout=DEFAULT_TIMEOUT):
        query_params = {"history_id": history_id}

        def get_jobs():
            jobs_response = self._get("jobs", query_params)
            assert jobs_response.status_code == 200
            return jobs_response.json()

        def has_active_jobs():
            jobs = get_jobs()
            active_jobs = [j for j in jobs if j["state"] in ["new", "upload", "waiting", "queued", "running"]]

            if len(active_jobs) == 0:
                return True
            else:
                return None

        try:
            wait_on(has_active_jobs, "active jobs", timeout=timeout)
        except TimeoutAssertionError as e:
            jobs = get_jobs()
            message = "Failed waiting on active jobs to complete, current jobs are [%s]. %s" % (jobs, e.message)
            raise TimeoutAssertionError(message)

        if assert_ok:
            jobs = get_jobs()
            hack_allow_new = True if len(jobs) > 0 else False
            return self.wait_for_history(history_id, assert_ok=True, timeout=timeout, hack_allow_new=hack_allow_new)

    def wait_for_job(self, job_id, assert_ok=False, timeout=DEFAULT_TIMEOUT):
        return wait_on_state(lambda: self.get_job_details(job_id), desc="job state", assert_ok=assert_ok, timeout=timeout)

    def get_job_details(self, job_id, full=False):
        return self._get("jobs/%s?full=%s" % (job_id, full))

    def cancel_job(self, job_id):
        return self._delete("jobs/%s" % job_id)

    def create_tool(self, representation, tool_directory=None):
        if isinstance(representation, dict):
            representation = json.dumps(representation)
        payload = dict(
            representation=representation,
            tool_directory=tool_directory,
        )
        try:
            create_response = self._post("dynamic_tools", data=payload, admin=True)
        except TypeError:
            create_response = self._post("dynamic_tools", data=payload)
        assert create_response.status_code == 200, create_response.json()
        return create_response.json()

    def _summarize_history(self, history_id):
        pass

    @contextlib.contextmanager
    def test_history(self, **kwds):
        # TODO: In the future allow targetting a specfic history here
        # and/or deleting everything in the resulting history when done.
        # These would be cool options for remote Galaxy test execution.
        try:
            history_id = self.new_history()
            yield history_id
        except Exception:
            self._summarize_history(history_id)
            raise

    def new_history(self, **kwds):
        name = kwds.get("name", "API Test History")
        create_history_response = self._post("histories", data=dict(name=name))
        assert "id" in create_history_response.json(), create_history_response.json()
        history_id = create_history_response.json()["id"]
        return history_id

    def upload_payload(self, history_id, content=None, **kwds):
        name = kwds.get("name", "Test_Dataset")
        dbkey = kwds.get("dbkey", "?")
        file_type = kwds.get("file_type", 'txt')
        upload_params = {
            'files_0|NAME': name,
            'dbkey': dbkey,
            'file_type': file_type,
        }
        if dbkey is None:
            del upload_params["dbkey"]
        if content is None:
            upload_params["files_0|ftp_files"] = kwds.get("ftp_files")
        elif hasattr(content, 'read'):
            upload_params["files_0|file_data"] = content
        else:
            upload_params['files_0|url_paste'] = content

        if "to_posix_lines" in kwds:
            upload_params["files_0|to_posix_lines"] = kwds["to_posix_lines"]
        if "space_to_tab" in kwds:
            upload_params["files_0|space_to_tab"] = kwds["space_to_tab"]
        if "auto_decompress" in kwds:
            upload_params["files_0|auto_decompress"] = kwds["auto_decompress"]
        upload_params.update(kwds.get("extra_inputs", {}))
        return self.run_tool_payload(
            tool_id='upload1',
            inputs=upload_params,
            history_id=history_id,
            upload_type='upload_dataset'
        )

    def get_remote_files(self, target="ftp"):
        return self._get("remote_files", data={"target": target}).json()

    def run_tool_payload(self, tool_id, inputs, history_id, **kwds):
        # Remove files_%d|file_data parameters from inputs dict and attach
        # as __files dictionary.
        for key, value in list(inputs.items()):
            if key.startswith("files_") and key.endswith("|file_data"):
                if "__files" not in kwds:
                    kwds["__files"] = {}
                kwds["__files"][key] = value
                del inputs[key]

        ir = kwds.get("inputs_representation", None)
        if ir is None and "inputs_representation" in kwds:
            del kwds["inputs_representation"]

        return dict(
            tool_id=tool_id,
            inputs=json.dumps(inputs),
            history_id=history_id,
            **kwds
        )

    def run_tool(self, tool_id, inputs, history_id, assert_ok=True, **kwds):
        payload = self.run_tool_payload(tool_id, inputs, history_id, **kwds)
        tool_response = self.tools_post(payload)
        if assert_ok:
            api_asserts.assert_status_code_is(tool_response, 200)
            return tool_response.json()
        else:
            return tool_response

    def tools_post(self, payload, url="tools"):
        tool_response = self._post(url, data=payload)
        return tool_response

    def get_history_dataset_content(self, history_id, wait=True, filename=None, **kwds):
        dataset_id = self.__history_content_id(history_id, wait=wait, **kwds)
        data = {}
        if filename:
            data["filename"] = filename
        display_response = self._get_contents_request(history_id, "/%s/display" % dataset_id, data=data)
        assert display_response.status_code == 200, display_response.content
        return display_response.content

    def get_history_dataset_details(self, history_id, **kwds):
        dataset_id = self.__history_content_id(history_id, **kwds)
        details_response = self._get_contents_request(history_id, "/datasets/%s" % dataset_id)
        assert details_response.status_code == 200
        return details_response.json()

    def get_history_dataset_extra_files(self, history_id, **kwds):
        dataset_id = self.__history_content_id(history_id, **kwds)
        details_response = self._get_contents_request(history_id, "/%s/extra_files" % dataset_id)
        assert details_response.status_code == 200, details_response.content
        return details_response.json()

    def get_history_collection_details(self, history_id, **kwds):
        hdca_id = self.__history_content_id(history_id, **kwds)
        details_response = self._get_contents_request(history_id, "/dataset_collections/%s" % hdca_id)
        assert details_response.status_code == 200, details_response.content
        return details_response.json()

    def run_collection_creates_list(self, history_id, hdca_id):
        inputs = {
            "input1": {"src": "hdca", "id": hdca_id},
        }
        self.wait_for_history(history_id, assert_ok=True)
        return self.run_tool("collection_creates_list", inputs, history_id)

    def run_exit_code_from_file(self, history_id, hdca_id):
        exit_code_inputs = {
            "input": {'batch': True, 'values': [{"src": "hdca", "id": hdca_id}]},
        }
        response = self.run_tool("exit_code_from_file", exit_code_inputs, history_id, assert_ok=False).json()
        self.wait_for_history(history_id, assert_ok=False)
        return response

    def __history_content_id(self, history_id, wait=True, **kwds):
        if wait:
            assert_ok = kwds.get("assert_ok", True)
            self.wait_for_history(history_id, assert_ok=assert_ok)
        # kwds should contain a 'dataset' object response, a 'dataset_id' or
        # the last dataset in the history will be fetched.
        if "dataset_id" in kwds:
            history_content_id = kwds["dataset_id"]
        elif "content_id" in kwds:
            history_content_id = kwds["content_id"]
        elif "dataset" in kwds:
            history_content_id = kwds["dataset"]["id"]
        else:
            hid = kwds.get("hid", None)  # If not hid, just grab last dataset
            history_contents = self._get_contents_request(history_id).json()
            if hid:
                history_content_id = None
                for history_item in history_contents:
                    if history_item["hid"] == hid:
                        history_content_id = history_item["id"]
                if history_content_id is None:
                    raise Exception("Could not find content with HID [%s] in [%s]" % (hid, history_contents))
            else:
                # No hid specified - just grab most recent element.
                history_content_id = history_contents[-1]["id"]
        return history_content_id

    def _get_contents_request(self, history_id, suffix="", data={}):
        url = "histories/%s/contents" % history_id
        if suffix:
            url = "%s%s" % (url, suffix)
        return self._get(url, data=data)

    def ds_entry(self, history_content):
        src = 'hda'
        if 'history_content_type' in history_content and history_content['history_content_type'] == "dataset_collection":
            src = 'hdca'
        return dict(src=src, id=history_content["id"])


class DatasetPopulator(BaseDatasetPopulator):

    def __init__(self, galaxy_interactor):
        self.galaxy_interactor = galaxy_interactor

    def _post(self, route, data={}, files=None, admin=False):
        files = data.get("__files", None)
        if files is not None:
            del data["__files"]

        return self.galaxy_interactor.post(route, data, files=files, admin=admin)

    def _get(self, route, data={}):
        return self.galaxy_interactor.get(route, data=data)

    def _delete(self, route, data={}):
        return self.galaxy_interactor.delete(route, data=data)

    def _summarize_history(self, history_id):
        self.galaxy_interactor._summarize_history(history_id)

    def wait_for_dataset(self, history_id, dataset_id, assert_ok=False, timeout=DEFAULT_TIMEOUT):
        return wait_on_state(lambda: self._get("histories/%s/contents/%s" % (history_id, dataset_id)), desc="dataset state", assert_ok=assert_ok, timeout=timeout)


class BaseWorkflowPopulator(object):

    def load_workflow(self, name, content=workflow_str, add_pja=False):
        workflow = json.loads(content)
        workflow["name"] = name
        if add_pja:
            tool_step = workflow["steps"]["2"]
            tool_step["post_job_actions"]["RenameDatasetActionout_file1"] = dict(
                action_type="RenameDatasetAction",
                output_name="out_file1",
                action_arguments=dict(newname="foo ${replaceme}"),
            )
        return workflow

    def load_random_x2_workflow(self, name):
        return self.load_workflow(name, content=workflow_random_x2_str)

    def load_workflow_from_resource(self, name, filename=None):
        if filename is None:
            filename = "data/%s.ga" % name
        content = resource_string(__name__, filename)
        return self.load_workflow(name, content=content)

    def simple_workflow(self, name, **create_kwds):
        workflow = self.load_workflow(name)
        return self.create_workflow(workflow, **create_kwds)

    def create_workflow(self, workflow, **create_kwds):
        upload_response = self.create_workflow_response(workflow, **create_kwds)
        uploaded_workflow_id = upload_response.json()["id"]
        return uploaded_workflow_id

    def create_workflow_response(self, workflow, **create_kwds):
        data = dict(
            workflow=json.dumps(workflow),
            **create_kwds
        )
        upload_response = self._post("workflows/upload", data=data)
        return upload_response

    def upload_yaml_workflow(self, has_yaml, **kwds):
        workflow = convert_and_import_workflow(has_yaml, galaxy_interface=self, **kwds)
        return workflow["id"]

    def wait_for_invocation(self, workflow_id, invocation_id, timeout=DEFAULT_TIMEOUT):
        url = "workflows/%s/usage/%s" % (workflow_id, invocation_id)
        return wait_on_state(lambda: self._get(url), desc="workflow invocation state", timeout=timeout)

    def wait_for_workflow(self, workflow_id, invocation_id, history_id, assert_ok=True, timeout=DEFAULT_TIMEOUT):
        """ Wait for a workflow invocation to completely schedule and then history
        to be complete. """
        self.wait_for_invocation(workflow_id, invocation_id, timeout=timeout)
        self.dataset_populator.wait_for_history_jobs(history_id, assert_ok=assert_ok, timeout=timeout)

    def invoke_workflow(self, history_id, workflow_id, inputs={}, request={}, assert_ok=True):
        request["history"] = "hist_id=%s" % history_id,
        if inputs:
            request["inputs"] = json.dumps(inputs)
            request["inputs_by"] = 'step_index'
        url = "workflows/%s/usage" % (workflow_id)
        invocation_response = self._post(url, data=request)
        if assert_ok:
            api_asserts.assert_status_code_is(invocation_response, 200)
            invocation_id = invocation_response.json()["id"]
            return invocation_id
        else:
            return invocation_response

    def wait_for_invocation_and_jobs(self, history_id, workflow_id, invocation_id, assert_ok=True):
        # TODO: replace instances in workflow test cases with this to de-dup...
        self.wait_for_invocation(workflow_id, invocation_id)
        url = "workflows/%s/usage/%s" % (workflow_id, invocation_id)
        invocation = self._get(url)
        steps = invocation.json()["steps"]
        wait_for_jobs = False
        for step in steps:
            if step.get("job_id"):
                wait_for_jobs = True
                break
        time.sleep(.05)
        if wait_for_jobs:
            self.dataset_populator.wait_for_history_jobs(history_id, assert_ok=assert_ok)
        time.sleep(.05)


class WorkflowPopulator(BaseWorkflowPopulator, ImporterGalaxyInterface):

    def __init__(self, galaxy_interactor):
        self.galaxy_interactor = galaxy_interactor
        self.dataset_populator = DatasetPopulator(galaxy_interactor)

    def _post(self, route, data={}, admin=False):
        return self.galaxy_interactor.post(route, data, admin=admin)

    def _get(self, route, data={}):
        return self.galaxy_interactor.get(route, data=data)

    # Required for ImporterGalaxyInterface interface - so we can recurisvely import
    # nested workflows.
    def import_workflow(self, workflow, **kwds):
        workflow_str = json.dumps(workflow, indent=4)
        data = {
            'workflow': workflow_str,
        }
        data.update(**kwds)
        upload_response = self._post("workflows", data=data)
        assert upload_response.status_code == 200, upload_response
        return upload_response.json()

    def import_tool(self, tool):
        """ Import a workflow via POST /api/workflows or
        comparable interface into Galaxy.
        """
        upload_response = self._import_tool_response(tool)
        assert upload_response.status_code == 200, upload_response
        return upload_response.json()

    def _import_tool_response(self, tool):
        tool_str = json.dumps(tool, indent=4)
        data = {
            'representation': tool_str
        }
        upload_response = self._post("dynamic_tools", data=data, admin=True)
        return upload_response


class LibraryPopulator(object):

    def __init__(self, galaxy_interactor):
        self.galaxy_interactor = galaxy_interactor
        self.dataset_populator = DatasetPopulator(galaxy_interactor)

    def get_libraries(self):
        get_response = self.galaxy_interactor.get("libraries")
        return get_response.json()

    def new_private_library(self, name):
        library = self.new_library(name)
        library_id = library["id"]

        role_id = self.user_private_role_id()
        self.set_permissions(library_id, role_id)
        return library

    def new_library(self, name):
        data = dict(name=name)
        create_response = self.galaxy_interactor.post("libraries", data=data, admin=True)
        return create_response.json()

    def set_permissions(self, library_id, role_id=None):
        if role_id:
            perm_list = json.dumps(role_id)
        else:
            perm_list = json.dumps([])

        permissions = dict(
            LIBRARY_ACCESS_in=perm_list,
            LIBRARY_MODIFY_in=perm_list,
            LIBRARY_ADD_in=perm_list,
            LIBRARY_MANAGE_in=perm_list,
        )
        self.galaxy_interactor.post("libraries/%s/permissions" % library_id, data=permissions, admin=True)

    def user_email(self):
        users_response = self.galaxy_interactor.get("users")
        users = users_response.json()
        assert len(users) == 1
        return users[0]["email"]

    def user_private_role_id(self):
        user_email = self.user_email()
        roles_response = self.galaxy_interactor.get("roles", admin=True)
        users_roles = [r for r in roles_response.json() if r["name"] == user_email]
        assert len(users_roles) == 1
        return users_roles[0]["id"]

    def create_dataset_request(self, library, **kwds):
        upload_option = kwds.get("upload_option", "upload_file")
        create_data = {
            "folder_id": kwds.get("folder_id", library["root_folder_id"]),
            "create_type": "file",
            "files_0|NAME": kwds.get("name", "NewFile"),
            "upload_option": upload_option,
            "file_type": kwds.get("file_type", "auto"),
            "db_key": kwds.get("db_key", "?"),
        }
        if kwds.get("link_data"):
            create_data["link_data_only"] = "link_to_files"

        if upload_option == "upload_file":
            files = {
                "files_0|file_data": kwds.get("file", StringIO(kwds.get("contents", "TestData"))),
            }
        elif upload_option == "upload_paths":
            create_data["filesystem_paths"] = kwds["paths"]
            files = {}
        elif upload_option == "upload_directory":
            create_data["server_dir"] = kwds["server_dir"]
            files = {}

        return create_data, files

    def new_library_dataset(self, name, **create_dataset_kwds):
        library = self.new_private_library(name)
        payload, files = self.create_dataset_request(library, **create_dataset_kwds)
        dataset = self.raw_library_contents_create(library["id"], payload, files=files).json()[0]
        return self.wait_on_library_dataset(library, dataset)

    def wait_on_library_dataset(self, library, dataset):
        def show():
            return self.galaxy_interactor.get("libraries/%s/contents/%s" % (library["id"], dataset["id"]))

        wait_on_state(show, assert_ok=True, timeout=DEFAULT_TIMEOUT)
        return show().json()

    def raw_library_contents_create(self, library_id, payload, files={}):
        url_rel = "libraries/%s/contents" % library_id
        return self.galaxy_interactor.post(url_rel, payload, files=files)

    def show_ldda(self, library_id, library_dataset_id):
        return self.galaxy_interactor.get("libraries/%s/contents/%s" % (library_id, library_dataset_id))

    def new_library_dataset_in_private_library(self, library_name="private_dataset", wait=True):
        library = self.new_private_library(library_name)
        payload, files = self.create_dataset_request(library, file_type="txt", contents="create_test")
        create_response = self.galaxy_interactor.post("libraries/%s/contents" % library["id"], payload, files=files)
        api_asserts.assert_status_code_is(create_response, 200)
        library_datasets = create_response.json()
        assert len(library_datasets) == 1
        library_dataset = library_datasets[0]
        if wait:
            def show():
                return self.show_ldda(library["id"], library_dataset["id"])

            wait_on_state(show, assert_ok=True)
            library_dataset = show().json()

        return library, library_dataset

    def get_library_contents_with_path(self, library_id, path):
        all_contents_response = self.galaxy_interactor.get("libraries/%s/contents" % library_id)
        api_asserts.assert_status_code_is(all_contents_response, 200)
        all_contents = all_contents_response.json()
        matching = [c for c in all_contents if c["name"] == path]
        if len(matching) == 0:
            raise Exception("Failed to find library contents with path [%s], contents are %s" % (path, all_contents))
        get_response = self.galaxy_interactor.get(matching[0]["url"])
        api_asserts.assert_status_code_is(get_response, 200)
        return get_response.json()

    def setup_fetch_to_folder(self, test_name):
        history_id = self.dataset_populator.new_history()
        library = self.new_private_library(test_name)
        folder_id = library["root_folder_id"][1:]
        destination = {"type": "library_folder", "library_folder_id": folder_id}
        return history_id, library, destination


class BaseDatasetCollectionPopulator(object):

    def create_list_from_pairs(self, history_id, pairs, name="Dataset Collection from pairs"):
        return self.create_nested_collection(history_id=history_id,
                                             collection=pairs,
                                             collection_type='list:paired',
                                             name=name)

    def nested_collection_identifiers(self, history_id, collection_type):
        rank_types = list(reversed(collection_type.split(":")))
        assert len(rank_types) > 0
        rank_type_0 = rank_types[0]
        if rank_type_0 == "list":
            identifiers = self.list_identifiers(history_id)
        else:
            identifiers = self.pair_identifiers(history_id)
        nested_collection_type = rank_type_0

        for i, rank_type in enumerate(reversed(rank_types[1:])):
            name = "test_level_%d" % (i + 1) if rank_type == "list" else "paired"
            identifiers = [dict(
                src="new_collection",
                name=name,
                collection_type=nested_collection_type,
                element_identifiers=identifiers,
            )]
            nested_collection_type = "%s:%s" % (rank_type, nested_collection_type)
        return identifiers

    def create_nested_collection(self, history_id, collection_type, name=None, collection=None, element_identifiers=None):
        """Create a nested collection either from collection or using collection_type)."""
        assert collection_type is not None
        name = name or "Test %s" % collection_type
        if collection is not None:
            assert element_identifiers is None
            element_identifiers = []
            for i, pair in enumerate(collection):
                element_identifiers.append(dict(
                    name="test%d" % i,
                    src="hdca",
                    id=pair
                ))
        if element_identifiers is None:
            element_identifiers = self.nested_collection_identifiers(history_id, collection_type)

        payload = dict(
            instance_type="history",
            history_id=history_id,
            element_identifiers=json.dumps(element_identifiers),
            collection_type=collection_type,
            name=name,
        )
        return self.__create(payload)

    def create_list_of_pairs_in_history(self, history_id, **kwds):
        pair1 = self.create_pair_in_history(history_id, **kwds).json()["id"]
        return self.create_list_from_pairs(history_id, [pair1])

    def create_list_of_list_in_history(self, history_id, **kwds):
        # create_nested_collection will generate nested collection from just datasets,
        # this function uses recursive generation of history hdcas.
        collection_type = kwds.pop('collection_type', 'list:list')
        collection_types = collection_type.split(':')
        list = self.create_list_in_history(history_id, **kwds).json()['id']
        current_collection_type = 'list'
        for collection_type in collection_types[1:]:
            current_collection_type = "%s:%s" % (current_collection_type, collection_type)
            response = self.create_nested_collection(history_id=history_id,
                                                     collection_type=current_collection_type,
                                                     name=current_collection_type,
                                                     collection=[list])
            list = response.json()['id']
        return response

    def create_pair_in_history(self, history_id, **kwds):
        payload = self.create_pair_payload(
            history_id,
            instance_type="history",
            **kwds
        )
        return self.__create(payload)

    def create_list_in_history(self, history_id, **kwds):
        payload = self.create_list_payload(
            history_id,
            instance_type="history",
            **kwds
        )
        return self.__create(payload)

    def create_list_payload(self, history_id, **kwds):
        return self.__create_payload(history_id, identifiers_func=self.list_identifiers, collection_type="list", **kwds)

    def create_pair_payload(self, history_id, **kwds):
        return self.__create_payload(history_id, identifiers_func=self.pair_identifiers, collection_type="paired", **kwds)

    def __create_payload(self, history_id, identifiers_func, collection_type, **kwds):
        contents = None
        if "contents" in kwds:
            contents = kwds["contents"]
            del kwds["contents"]

        if "element_identifiers" not in kwds:
            kwds["element_identifiers"] = json.dumps(identifiers_func(history_id, contents=contents))

        if "name" not in kwds:
            kwds["name"] = "Test Dataset Collection"

        payload = dict(
            history_id=history_id,
            collection_type=collection_type,
            **kwds
        )
        return payload

    def pair_identifiers(self, history_id, contents=None):
        hda1, hda2 = self.__datasets(history_id, count=2, contents=contents)

        element_identifiers = [
            dict(name="forward", src="hda", id=hda1["id"]),
            dict(name="reverse", src="hda", id=hda2["id"]),
        ]
        return element_identifiers

    def list_identifiers(self, history_id, contents=None):
        count = 3 if contents is None else len(contents)
        # Contents can be a list of strings (with name auto-assigned here) or a list of
        # 2-tuples of form (name, dataset_content).
        if contents and isinstance(contents[0], tuple):
            hdas = self.__datasets(history_id, count=count, contents=[c[1] for c in contents])

            def hda_to_identifier(i, hda):
                return dict(name=contents[i][0], src="hda", id=hda["id"])
        else:
            hdas = self.__datasets(history_id, count=count, contents=contents)

            def hda_to_identifier(i, hda):
                return dict(name="data%d" % (i + 1), src="hda", id=hda["id"])
        element_identifiers = [hda_to_identifier(i, hda) for (i, hda) in enumerate(hdas)]
        return element_identifiers

    def __create(self, payload):
        return self._create_collection(payload)

    def __datasets(self, history_id, count, contents=None):
        datasets = []
        for i in range(count):
            new_kwds = {}
            if contents:
                new_kwds["content"] = contents[i]
            datasets.append(self.dataset_populator.new_dataset(history_id, **new_kwds))
        return datasets

    def wait_for_dataset_collection(self, create_payload, assert_ok=False, timeout=DEFAULT_TIMEOUT):
        for element in create_payload["elements"]:
            if element['element_type'] == 'hda':
                self.dataset_populator.wait_for_dataset(history_id=element['object']['history_id'],
                                                        dataset_id=element['object']['id'],
                                                        assert_ok=assert_ok,
                                                        timeout=timeout)
            elif element['element_type'] == 'dataset_collection':
                self.wait_for_dataset_collection(element['object'], assert_ok=assert_ok, timeout=timeout)


class DatasetCollectionPopulator(BaseDatasetCollectionPopulator):

    def __init__(self, galaxy_interactor):
        self.galaxy_interactor = galaxy_interactor
        self.dataset_populator = DatasetPopulator(galaxy_interactor)

    def _create_collection(self, payload):
        create_response = self.galaxy_interactor.post("dataset_collections", data=payload)
        return create_response


def load_data_dict(history_id, test_data, dataset_populator, dataset_collection_populator):

    def read_test_data(test_dict):
        test_data_resolver = TestDataResolver()
        filename = test_data_resolver.get_filename(test_dict["value"])
        content = open(filename, "r").read()
        return content

    inputs = {}
    label_map = {}
    has_uploads = False

    for key, value in test_data.items():
        is_dict = isinstance(value, dict)
        if is_dict and ("elements" in value or value.get("type", None) in ["list:paired", "list", "paired"]):
            elements_data = value.get("elements", [])
            elements = []
            for element_data in elements_data:
                identifier = element_data["identifier"]
                input_type = element_data.get("type", "raw")
                if input_type == "File":
                    content = read_test_data(element_data)
                else:
                    content = element_data["content"]
                elements.append((identifier, content))
            # TODO: make this collection_type
            collection_type = value["type"]
            new_collection_kwds = {}
            if "name" in value:
                new_collection_kwds["name"] = value["name"]
            if collection_type == "list:paired":
                hdca = dataset_collection_populator.create_list_of_pairs_in_history(history_id, **new_collection_kwds).json()
            elif collection_type == "list":
                hdca = dataset_collection_populator.create_list_in_history(history_id, contents=elements, **new_collection_kwds).json()
            else:
                hdca = dataset_collection_populator.create_pair_in_history(history_id, contents=elements, **new_collection_kwds).json()
            label_map[key] = dataset_populator.ds_entry(hdca)
            inputs[key] = hdca
            has_uploads = True
        elif is_dict and "type" in value:
            input_type = value["type"]
            if input_type == "File":
                content = read_test_data(value)
                new_dataset_kwds = {
                    "content": content
                }
                if "name" in value:
                    new_dataset_kwds["name"] = value["name"]
                if "file_type" in value:
                    new_dataset_kwds["file_type"] = value["file_type"]
                hda = dataset_populator.new_dataset(history_id, **new_dataset_kwds)
                label_map[key] = dataset_populator.ds_entry(hda)
                has_uploads = True
            elif input_type == "raw":
                label_map[key] = value["value"]
                inputs[key] = value["value"]
        elif not is_dict:
            has_uploads = True
            hda = dataset_populator.new_dataset(history_id, content=value)
            label_map[key] = dataset_populator.ds_entry(hda)
            inputs[key] = hda
        else:
            raise ValueError("Invalid test_data def %" % test_data)

    return inputs, label_map, has_uploads


def wait_on_state(state_func, desc="state", skip_states=["running", "queued", "new", "ready"], ok_states=["ok"], assert_ok=False, timeout=DEFAULT_TIMEOUT):
    def get_state():
        response = state_func()
        assert response.status_code == 200, "Failed to fetch state update while waiting."
        state = response.json()["state"]
        if state in skip_states:
            return None
        else:
            if assert_ok:
                assert state in ok_states, "Final state - %s - not okay." % state
            return state
    try:
        return wait_on(get_state, desc=desc, timeout=timeout)
    except TimeoutAssertionError as e:
        response = state_func()
        raise TimeoutAssertionError("%s Current response containing state [%s]." % (e.message, response.json()))


class GiPostGetMixin:
    """Mixin for adapting Galaxy testing populators helpers to bioblend."""

    def _get(self, route, data={}):
        return self._gi.make_get_request(self.__url(route), data=data)

    def _post(self, route, data={}):
        data = data.copy()
        data['key'] = self._gi.key
        return requests.post(self.__url(route), data=data)

    def _delete(self, route, data={}):
        data = data.copy()
        data['key'] = self._gi.key
        return requests.delete(self.__url(route), data=data)

    def __url(self, route):
        return self._gi.url + "/" + route


class GiDatasetPopulator(BaseDatasetPopulator, GiPostGetMixin):

    """Implementation of BaseDatasetPopulator backed by bioblend."""

    def __init__(self, gi):
        """Construct a dataset populator from a bioblend GalaxyInstance."""
        self._gi = gi


class GiDatasetCollectionPopulator(BaseDatasetCollectionPopulator, GiPostGetMixin):

    """Implementation of BaseDatasetCollectionPopulator backed by bioblend."""

    def __init__(self, gi):
        """Construct a dataset collection populator from a bioblend GalaxyInstance."""
        self._gi = gi
        self.dataset_populator = GiDatasetPopulator(gi)

    def _create_collection(self, payload):
        create_response = self._post("dataset_collections", data=payload)
        return create_response


class GiWorkflowPopulator(BaseWorkflowPopulator, GiPostGetMixin):

    """Implementation of BaseWorkflowPopulator backed by bioblend."""

    def __init__(self, gi):
        """Construct a workflow populator from a bioblend GalaxyInstance."""
        self._gi = gi
        self.dataset_populator = GiDatasetPopulator(gi)


def wait_on(function, desc, timeout=DEFAULT_TIMEOUT):
    delta = .25
    iteration = 0
    while True:
        total_wait = delta * iteration
        if total_wait > timeout:
            timeout_message = "Timed out after %s seconds waiting on %s." % (
                total_wait, desc
            )
            raise TimeoutAssertionError(timeout_message)
        iteration += 1
        value = function()
        if value is not None:
            return value
        time.sleep(delta)


class TimeoutAssertionError(AssertionError):

    def __init__(self, message):
        super(TimeoutAssertionError, self).__init__(message)
