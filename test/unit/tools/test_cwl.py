
import os
import sys
import tempfile
import shutil

from unittest import TestCase

import galaxy.model

from galaxy.tools.cwl import tool_proxy
from galaxy.tools.cwl.parser import ToolProxy
from galaxy.tools.cwl import workflow_proxy

from galaxy.tools.parser.factory import get_tool_source

import tools_support

unit_root = os.path.abspath( os.path.join( os.path.dirname( __file__ ), os.pardir ) )
sys.path.insert( 1, unit_root )
from unittest_utils import galaxy_mock

TESTS_DIRECTORY = os.path.dirname(__file__)
CWL_TOOLS_DIRECTORY = os.path.abspath(os.path.join(TESTS_DIRECTORY, "cwl_tools"))


def test_tool_proxy():
    """Test that tool proxies load some valid tools correctly."""
    tool_proxy(_cwl_tool_path("draft3/cat1-tool.cwl"))
    tool_proxy(_cwl_tool_path("draft3/cat3-tool.cwl"))
    tool_proxy(_cwl_tool_path("draft3/env-tool1.cwl"))
    tool_proxy(_cwl_tool_path("draft3/sorttool.cwl"))
    tool_proxy(_cwl_tool_path("draft3/bwa-mem-tool.cwl"))

    tool_proxy(_cwl_tool_path("draft3/parseInt-tool.cwl"))


def test_tool_source_records():
    record_output_path = _cwl_tool_path("v1.0/record-output.cwl")
    tool_source = get_tool_source(record_output_path)
    inputs = _inputs(tool_source)
    assert len(inputs) == 1

    output_data, output_collections = _outputs(tool_source)
    assert len(output_data) == 1
    assert len(output_collections) == 1


def test_serialize_deserialize():
    tool = tool_proxy(_cwl_tool_path("draft3/cat1-tool.cwl"))
    ToolProxy.from_persistent_representation(tool.to_persistent_representation())


def test_reference_proxies():
    versions = ["draft3", "v1.0"]
    for version in versions:
        proxy = workflow_proxy(_cwl_tool_path("%s/count-lines1-wf.cwl" % version))
        proxy.tool_reference_proxies()


def test_checks_requirements():
    """Test that tool proxy will not load unmet requirements."""
    exception = None
    try:
        tool_proxy(_cwl_tool_path("draft3_custom/unmet-requirement.cwl"))
    except Exception as e:
        exception = e

    assert exception is not None
    assert "Unsupported requirement SubworkflowFeatureRequirement" in str(exception), str(exception)


def test_checks_is_a_tool():
    """Test that tool proxy cannot be created for a workflow."""
    exception = None
    try:
        tool_proxy(_cwl_tool_path("draft3/count-lines1-wf.cwl"))
    except Exception as e:
        exception = e

    assert exception is not None
    assert "CommandLineTool" in str(exception), str(exception)


def test_checks_cwl_version():
    """Test that tool proxy verifies supported version."""
    exception = None
    try:
        tool_proxy(_cwl_tool_path("draft3_custom/version345.cwl"))
    except Exception as e:
        exception = e

    assert exception is not None


def test_workflow_of_files_proxy():
    versions = ["draft3", "v1.0"]
    for version in versions:
        proxy = workflow_proxy(_cwl_tool_path("%s/count-lines1-wf.cwl" % version))
        step_proxies = proxy.step_proxies()
        assert len(step_proxies) == 2

        galaxy_workflow_dict = proxy.to_dict()

        assert len(proxy.runnables) == 2

        assert len(galaxy_workflow_dict["steps"]) == 3
        wc_step = galaxy_workflow_dict["steps"][1]
        exp_step = galaxy_workflow_dict["steps"][2]
        assert wc_step["input_connections"]
        assert exp_step["input_connections"]


def test_workflow_embedded_tools_proxy():
    versions = ["draft3", "v1.0"]
    for version in versions:
        proxy = workflow_proxy(_cwl_tool_path("%s/count-lines2-wf.cwl" % version))
        step_proxies = proxy.step_proxies()
        assert len(step_proxies) == 2

        galaxy_workflow_dict = proxy.to_dict()

        assert len(proxy.runnables) == 2

        assert len(galaxy_workflow_dict["steps"]) == 3
        wc_step = galaxy_workflow_dict["steps"][1]
        exp_step = galaxy_workflow_dict["steps"][2]
        assert wc_step["input_connections"]
        assert exp_step["input_connections"]


def test_workflow_scatter():
    version = "v1.0"
    proxy = workflow_proxy(_cwl_tool_path("%s/count-lines3-wf.cwl" % version))

    step_proxies = proxy.step_proxies()
    assert len(step_proxies) == 1

    galaxy_workflow_dict = proxy.to_dict()
    assert len(galaxy_workflow_dict["steps"]) == 2

    # TODO: For CWL - deactivate implicit scattering Galaxy does
    # and force annotation in the workflow of scattering? Maybe?
    wc_step = galaxy_workflow_dict["steps"][1]
    assert wc_step["input_connections"]


def test_workflow_scatter_multiple_input():
    version = "v1.0"
    proxy = workflow_proxy(_cwl_tool_path("%s/count-lines4-wf.cwl" % version))

    step_proxies = proxy.step_proxies()
    assert len(step_proxies) == 1

    galaxy_workflow_dict = proxy.to_dict()
    assert len(galaxy_workflow_dict["steps"]) == 3


def test_load_proxy_simple():
    cat3 = _cwl_tool_path("draft3/cat3-tool.cwl")
    tool_source = get_tool_source(cat3)

    description = tool_source.parse_description()
    assert description == "Print the contents of a file to stdout using 'cat' running in a docker container.", description

    input_sources = _inputs(tool_source)
    assert len(input_sources) == 1

    input_source = input_sources[0]
    assert input_source.parse_help() == "The file that will be copied using 'cat'"
    assert input_source.parse_label() == "Input File"

    outputs, output_collections = tool_source.parse_outputs(None)
    assert len(outputs) == 1

    output1 = outputs['output_file']
    assert output1.format == "_sniff_"  # Have Galaxy auto-detect

    _, containers = tool_source.parse_requirements_and_containers()
    assert len(containers) == 1


def test_cwl_strict_parsing():
    md5sum_non_strict_path = _cwl_tool_path("v1.0_custom/md5sum_non_strict.cwl")
    threw_exception = False
    try:
        get_tool_source(md5sum_non_strict_path).tool_proxy
    except Exception:
        threw_exception = True

    assert threw_exception
    get_tool_source(md5sum_non_strict_path, strict_cwl_validation=False).tool_proxy


def test_load_proxy_bwa_mem():
    bwa_mem = _cwl_tool_path("draft3/bwa-mem-tool.cwl")
    tool_source = get_tool_source(bwa_mem)
    tool_id = tool_source.parse_id()
    assert tool_id == "bwa-mem-tool", tool_id
    _inputs(tool_source)
    # TODO: test repeat generated...


def test_env_tool1():
    env_tool1 = _cwl_tool_path("draft3/env-tool1.cwl")
    tool_source = get_tool_source(env_tool1)
    _inputs(tool_source)


def test_wc2_tool():
    env_tool1 = _cwl_tool_path("draft3/wc2-tool.cwl")
    tool_source = get_tool_source(env_tool1)
    _inputs(tool_source)
    datasets, collections = _outputs(tool_source)
    assert len(datasets) == 1, datasets
    output = datasets["output"]
    assert output.format == "expression.json", output.format


def test_optionial_output2():
    optional_output2_tool1 = _cwl_tool_path("draft3_custom/optional-output2.cwl")
    tool_source = get_tool_source(optional_output2_tool1)
    datasets, collections = _outputs(tool_source)
    assert len(datasets) == 1, datasets
    output = datasets["optional_file"]
    assert output.format == "_sniff_", output.format


def test_sorttool():
    env_tool1 = _cwl_tool_path("draft3/sorttool.cwl")
    tool_source = get_tool_source(env_tool1)

    assert tool_source.parse_id() == "sorttool"

    inputs = _inputs(tool_source)
    assert len(inputs) == 2
    bool_input = inputs[0]
    file_input = inputs[1]
    assert bool_input.parse_input_type() == "param"
    assert bool_input.get("type") == "boolean"

    assert file_input.parse_input_type() == "param"
    assert file_input.get("type") == "data", file_input.get("type")

    output_data, output_collections = _outputs(tool_source)
    assert len(output_data) == 1
    assert len(output_collections) == 0


def test_cat1():
    cat1_tool = _cwl_tool_path("draft3/cat1-tool.cwl")
    tool_source = get_tool_source(cat1_tool)
    inputs = _inputs(tool_source)

    assert len(inputs) == 2
    file_input = inputs[0]
    null_or_bool_input = inputs[1]

    assert file_input.parse_input_type() == "param"
    assert file_input.get("type") == "data", file_input.get("type")

    # User needs to specify if want to select boolean or not.
    assert null_or_bool_input.parse_input_type() == "conditional"

    output_data, output_collections = _outputs(tool_source)
    assert len(output_data) == 0
    assert len(output_collections) == 0


def test_tool_reload():
    cat1_tool = _cwl_tool_path("draft3/cat1-tool.cwl")
    tool_source = get_tool_source(cat1_tool)
    _inputs(tool_source)

    # Test reloading - had a regression where this broke down.
    cat1_tool_again = _cwl_tool_path("draft3/cat1-tool.cwl")
    tool_source = get_tool_source(cat1_tool_again)
    _inputs(tool_source)


class CwlToolObjectTestCase( TestCase, tools_support.UsesApp, tools_support.UsesTools ):

    def setUp(self):
        self.test_directory = tempfile.mkdtemp()
        self.app = galaxy_mock.MockApp()
        self.history = galaxy.model.History()
        self.trans = galaxy_mock.MockTrans( history=self.history )

    def tearDown( self ):
        shutil.rmtree( self.test_directory )


    def test_default_data_inputs(self):
        self._init_tool(tool_path=_cwl_tool_path("v1.0/default_path.cwl"))
        print("TOOL IS %s" % self.tool)
        hda = self._new_hda()
        from galaxy.tools.cwl import to_cwl_job
        from galaxy.tools.parameters import populate_state
        errors = {}
        cwl_inputs = {
            "file1": {"src": "hda", "id": self.app.security.encode_id(hda.id)}
        }
        inputs = self.tool.inputs_from_dict({"inputs": cwl_inputs, "inputs_representation": "cwl"})
        print inputs
        print("pre-populated state is %s" % inputs)
        populated_state = {}
        populate_state(self.trans, self.tool.inputs, inputs, populated_state, errors)
        print("populated state is %s" % inputs)
        from galaxy.tools.parameters.wrapped import WrappedParameters
        wrapped_params = WrappedParameters(galaxy_mock.MockTrans(), self.tool, populated_state)
        input_json = to_cwl_job(self.tool, wrapped_params.params, self.test_directory)
        print inputs
        print "to_cwl_job is %s" % input_json
        assert False

    def _new_hda( self ):
        hda = galaxy.model.HistoryDatasetAssociation(history=self.history)
        hda.visible = True
        hda.dataset = galaxy.model.Dataset()
        self.app.model.context.add( hda )
        self.app.model.context.flush( )
        return hda


def _outputs(tool_source):
    return tool_source.parse_outputs(object())


def get_cwl_tool_source(path):
    path = _cwl_tool_path(path)
    return get_tool_source(path)


def _inputs(tool_source=None, path=None):
    if tool_source is None:
        path = _cwl_tool_path(path)
        tool_source = get_tool_source(path)

    input_pages = tool_source.parse_input_pages()
    assert input_pages.inputs_defined
    page_sources = input_pages.page_sources
    assert len(page_sources) == 1
    page_source = page_sources[0]
    input_sources = page_source.parse_input_sources()
    return input_sources


def _cwl_tool_path(path):
    return os.path.join(CWL_TOOLS_DIRECTORY, path)
