import os
import string
import sys

import yaml

THIS_DIRECTORY = os.path.dirname(os.path.normpath(__file__))
API_TEST_DIRECTORY = os.path.join(THIS_DIRECTORY, "..", "..", "..", "api")

TEST_FILE_TEMPLATE = string.Template('''
"""Test CWL conformance for version $version."""

from .test_workflows_cwl import BaseCwlWorklfowTestCase


class CwlConformanceTestCase(BaseCwlWorklfowTestCase):
    """Test case mapping to CWL conformance tests for version $version."""
$tests
''')

TEST_TEMPLATE = string.Template('''
    def test_conformance_${version_simple}_${label}(self):
        """${doc}

        Generated from::

${cwl_test_def}
        """
        self.cwl_populator.run_conformance_test("""${version}""", """${doc}""")
''')

RED_TESTS = {
    "cl_basic_generation": "resource allocation mapping not implemented",
}


GREEN_TESTS = [
        "170",
        "cl_gen_arrayofarrays",
        "directory_output",
        "docker_json_output_location",
        "docker_json_output_path",
        "envvar_req",
        "expression_any",
        "expression_any_null",
        "expression_outputEval",
        "expression_parseint",
        "exprtool_directory_literal",
        "exprtool_file_literal",
        "initial_workdir_empty_writable",
        "initial_workdir_empty_writable_docker",
        "initial_workdir_expr",
        "initial_workdir_output",
        "initial_workdir_trailingnl",
        "initialworkpath_output",
        "inline_expressions",
        "metadata",
        "nameroot_nameext_stdout_expr",
        "null_missing_params",
        "rename",
        "stdinout_redirect",
        "stdinout_redirect_docker",
        "stdout_redirect_docker",
        "stdout_redirect_mediumcut_docker",
        "stdout_redirect_shortcut_docker",
        "writable_stagedfiles",
]

GREEN_TESTS += [
    "cl_optional_inputs_missing",
    "cl_optional_bindings_provided",
    "initworkdir_expreng_requirements",
    "nested_cl_bindings",
    "nested_prefixes_arrays",
    "stderr_redirect",
]


def main():
    version = "v1.0"
    if len(sys.argv) > 1:
        version = sys.argv[1]
    version_simple = version.replace(".", "_")
    conformance_tests_path = os.path.join(THIS_DIRECTORY, version, "conformance_tests.yaml")
    with open(conformance_tests_path, "r") as f:
        conformance_tests = yaml.load(f)

    tests = ""
    green_tests = ""
    for i, conformance_test in enumerate(conformance_tests):
        test_with_doc = conformance_test.copy()
        del test_with_doc["doc"]
        cwl_test_def = yaml.dump(test_with_doc, default_flow_style=False)
        cwl_test_def = "\n".join(["            %s" % l for l in cwl_test_def.splitlines()])
        label = conformance_test.get("label", str(i))
        tests = tests + TEST_TEMPLATE.safe_substitute({
            'version_simple': version_simple,
            'version': version,
            'doc': conformance_test['doc'],
            'cwl_test_def': cwl_test_def,
            'label': label,
        })
        if label in GREEN_TESTS:
            green_tests = green_tests + TEST_TEMPLATE.safe_substitute({
                'version_simple': version_simple,
                'version': version,
                'doc': conformance_test['doc'],
                'cwl_test_def': cwl_test_def,
                'label': label,
            })

    test_file_contents = TEST_FILE_TEMPLATE.safe_substitute({
        'version_simple': version_simple,
        'tests': tests
    })

    green_test_file_contents = TEST_FILE_TEMPLATE.safe_substitute({
        'version_simple': version_simple,
        'tests': green_tests
    })

    with open(os.path.join(API_TEST_DIRECTORY, "test_cwl_conformance_%s.py" % version_simple), "w") as f:
        f.write(test_file_contents)
    with open(os.path.join(API_TEST_DIRECTORY, "test_cwl_conformance_green_%s.py" % version_simple), "w") as f:
        f.write(green_test_file_contents)


if __name__ == "__main__":
    main()
