""" This module provides proxy objects around objects from the common
workflow language reference implementation library cwltool. These proxies
adapt cwltool to Galaxy features and abstract the library away from the rest
of the framework.
"""
from __future__ import absolute_import

import base64
import json
import logging
import os
import pickle
from abc import ABCMeta, abstractmethod

import six

from galaxy.tools.hash import build_tool_hash
from galaxy.util import listify, safe_makedirs
from galaxy.util.bunch import Bunch
from galaxy.util.odict import odict

from .cwltool_deps import (
    ensure_cwltool_available,
    load_tool,
    process,
    workflow,
)
from .representation import field_to_field_type, INPUT_TYPE, type_descriptions_for_field_types

from .schema import non_strict_schema_loader, schema_loader

log = logging.getLogger(__name__)

JOB_JSON_FILE = ".cwl_job.json"
SECONDARY_FILES_EXTRA_PREFIX = "__secondary_files__"

DOCKER_REQUIREMENT = "DockerRequirement"
SUPPORTED_TOOL_REQUIREMENTS = [
    "CreateFileRequirement",
    "DockerRequirement",
    "EnvVarRequirement",
    "InlineJavascriptRequirement",
    "ShellCommandRequirement",
    "ScatterFeatureRequirement",
    "MultipleInputFeatureRequirement",
]


SUPPORTED_WORKFLOW_REQUIREMENTS = SUPPORTED_TOOL_REQUIREMENTS + [
]


def tool_proxy(tool_path, strict_cwl_validation=True):
    """ Provide a proxy object to cwltool data structures to just
    grab relevant data.
    """
    ensure_cwltool_available()
    tool = to_cwl_tool_object(tool_path, strict_cwl_validation=strict_cwl_validation)
    return tool


def tool_proxy_from_persistent_representation(persisted_tool, strict_cwl_validation=True):
    ensure_cwltool_available()
    tool = to_cwl_tool_object(persisted_tool=persisted_tool, strict_cwl_validation=strict_cwl_validation)
    return tool


def workflow_proxy(workflow_path, strict_cwl_validation=True):
    ensure_cwltool_available()
    workflow = to_cwl_workflow_object(workflow_path, strict_cwl_validation=strict_cwl_validation)
    return workflow


def load_job_proxy(job_directory, strict_cwl_validation=True):
    ensure_cwltool_available()
    job_objects_path = os.path.join(job_directory, JOB_JSON_FILE)
    job_objects = json.load(open(job_objects_path, "r"))
    job_inputs = job_objects["job_inputs"]
    output_dict = job_objects["output_dict"]
    # Any reason to retain older tool_path variant of this? Probably not?
    if "tool_path" in job_objects:
        tool_path = job_objects["tool_path"]
        cwl_tool = tool_proxy(tool_path, strict_cwl_validation=strict_cwl_validation)
    else:
        persisted_tool = job_objects["tool_representation"]
        cwl_tool = tool_proxy_from_persistent_representation(persisted_tool, strict_cwl_validation=strict_cwl_validation)
    cwl_job = cwl_tool.job_proxy(job_inputs, output_dict, job_directory=job_directory)
    return cwl_job


def to_cwl_tool_object(tool_path=None, tool_object=None, persisted_tool=None, strict_cwl_validation=True):
    if tool_path is not None:
        cwl_tool = _schema_loader(strict_cwl_validation).tool(
            path=tool_path
        )
    else:
        cwl_tool = ToolProxy.from_persistent_representation(persisted_tool)

    if isinstance(cwl_tool, int):
        raise Exception("Failed to load tool.")

    raw_tool = cwl_tool.tool
    # Apply Galaxy hacks to CWL tool representation to bridge semantic differences
    # between Galaxy and cwltool.
    _hack_cwl_requirements(cwl_tool)
    check_requirements(raw_tool)
    return cwl_tool_object_to_proxy(cwl_tool, tool_path=tool_path)


def cwl_tool_object_to_proxy(cwl_tool, tool_path=None):
    raw_tool = cwl_tool.tool
    if "class" not in raw_tool:
        raise Exception("File does not declare a class, not a valid Draft 3+ CWL tool.")

    process_class = raw_tool["class"]
    if process_class == "CommandLineTool":
        proxy_class = CommandLineToolProxy
    elif process_class == "ExpressionTool":
        proxy_class = ExpressionToolProxy
    else:
        raise Exception("File not a CWL CommandLineTool.")
    top_level_object = tool_path is not None
    if top_level_object and ("cwlVersion" not in raw_tool):
        # cwl_tool.metadata["cwlVersion"]
        raise Exception("File does not declare a CWL version, pre-draft 3 CWL tools are not supported.")

    proxy = proxy_class(cwl_tool, tool_path)
    return proxy


def to_cwl_workflow_object(workflow_path, strict_cwl_validation=None):
    proxy_class = WorkflowProxy
    cwl_workflow = _schema_loader(strict_cwl_validation).tool(path=workflow_path)
    raw_workflow = cwl_workflow.tool
    check_requirements(raw_workflow, tool=False)

    proxy = proxy_class(cwl_workflow, workflow_path)
    return proxy


def _schema_loader(strict_cwl_validation):
    target_schema_loader = schema_loader if strict_cwl_validation else non_strict_schema_loader
    return target_schema_loader


def _hack_cwl_requirements(cwl_tool):
    raw_tool = cwl_tool.tool
    if "requirements" in raw_tool:
        requirements = raw_tool["requirements"]
        move_to_hint = None
        for i, r in enumerate(requirements):
            if r["class"] == DOCKER_REQUIREMENT:
                move_to_hint = i
        if move_to_hint is not None:
            hint = requirements.pop(move_to_hint)
            if "hints" not in raw_tool:
                raw_tool["hints"] = []
            raw_tool["hints"].append(hint)
    cwl_tool.requirements = raw_tool.get("requirements", [])


def check_requirements(rec, tool=True):
    if isinstance(rec, dict):
        if "requirements" in rec:
            for r in rec["requirements"]:
                if tool:
                    possible = SUPPORTED_TOOL_REQUIREMENTS
                else:
                    possible = SUPPORTED_WORKFLOW_REQUIREMENTS
                if r["class"] not in possible:
                    raise Exception("Unsupported requirement %s" % r["class"])
        for d in rec:
            check_requirements(rec[d], tool=tool)
    if isinstance(rec, list):
        for d in rec:
            check_requirements(d, tool=tool)


@six.add_metaclass(ABCMeta)
class ToolProxy( object ):

    def __init__(self, tool, tool_path=None):
        self._tool = tool
        self._tool_path = tool_path

    def job_proxy(self, input_dict, output_dict, job_directory="."):
        """ Build a cwltool.job.Job describing computation using a input_json
        Galaxy will generate mapping the Galaxy description of the inputs into
        a cwltool compatible variant.
        """
        return JobProxy(self, input_dict, output_dict, job_directory=job_directory)

    @property
    def id(self):
        print dir(self._tool.metadata)
        raw_id = self._tool.tool.get("id", None)
        return raw_id

    def galaxy_id(self):
        raw_id = self.id
        if raw_id:
            return os.path.splitext(os.path.basename(raw_id))[0]
        else:
            return build_tool_hash(self.to_persistent_representation())

    @abstractmethod
    def input_instances(self):
        """ Return InputInstance objects describing mapping to Galaxy inputs. """

    @abstractmethod
    def output_instances(self):
        """ Return OutputInstance objects describing mapping to Galaxy inputs. """

    @abstractmethod
    def docker_identifier(self):
        """ Return docker identifier for embedding in tool description. """

    @abstractmethod
    def description(self):
        """ Return description to tool. """

    @abstractmethod
    def label(self):
        """ Return label for tool. """

    def to_persistent_representation(self):
        """Return a JSON representation of this tool. Not for serialization
        over the wire, but serialization in a database."""
        # TODO: Replace this with some more readable serialization,
        # I really don't like using pickle here.
        return {
            "class": self._class,
            "pickle": base64.b64encode(pickle.dumps(remove_pickle_problems(self._tool), -1)),
        }

    @staticmethod
    def from_persistent_representation(as_object):
        """Recover an object serialized with to_persistent_representation."""
        if "class" not in as_object:
            raise Exception("Failed to deserialize tool proxy from JSON object - no class found.")
        if "pickle" not in as_object:
            raise Exception("Failed to deserialize tool proxy from JSON object - no pickle representation found.")
        return pickle.loads(base64.b64decode(as_object["pickle"]))


class CommandLineToolProxy(ToolProxy):
    _class = "CommandLineTool"

    def description(self):
        return self._tool.tool.get('doc')

    def label(self):
        return self._tool.tool.get('label')

    def input_fields(self):
        input_records_schema = self._tool.inputs_record_schema
        schema_type = input_records_schema["type"]
        if schema_type in self._tool.schemaDefs:
            input_records_schema = self._tool.schemaDefs[schema_type]

        if input_records_schema["type"] != "record":
            raise Exception("Unhandled CWL tool input structure")

        return input_records_schema["fields"]

    def input_instances(self):
        return [_outer_field_to_input_instance(_) for _ in self.input_fields()]

    def output_instances(self):
        outputs_schema = self._tool.outputs_record_schema
        schema_type = outputs_schema["type"]
        if schema_type in self._tool.schemaDefs:
            outputs_schema = self._tool.schemaDefs[schema_type]

        if outputs_schema["type"] != "record":
            raise Exception("Unhandled CWL tool output structure")

        rval = []
        for output in outputs_schema["fields"]:
            rval.append(_simple_field_to_output(output))

        return rval

    def docker_identifier(self):
        tool = self._tool.tool
        reqs_and_hints = tool.get("requirements", []) + tool.get("hints", [])
        for hint in reqs_and_hints:
            if hint["class"] == "DockerRequirement":
                if "dockerImageId" in hint:
                    return hint["dockerImageId"]
                else:
                    return hint["dockerPull"]
        return None


class ExpressionToolProxy(CommandLineToolProxy):
    _class = "ExpressionTool"


class JobProxy(object):

    def __init__(self, tool_proxy, input_dict, output_dict, job_directory):
        self._tool_proxy = tool_proxy
        self._input_dict = input_dict
        self._output_dict = output_dict
        self._job_directory = job_directory

        self._final_output = []
        self._ok = True
        self._cwl_job = None
        self._is_command_line_job = None

        self._normalize_job()

    def cwl_job(self):
        self._ensure_cwl_job_initialized()
        return self._cwl_job

    @property
    def is_command_line_job(self):
        self._ensure_cwl_job_initialized()
        assert self._is_command_line_job is not None
        return self._is_command_line_job

    def _ensure_cwl_job_initialized(self):
        if self._cwl_job is None:

            self._cwl_job = next(self._tool_proxy._tool.job(
                self._input_dict,
                self._output_callback,
                basedir=self._job_directory,
                select_resources=self._select_resources,
                outdir=os.path.join(self._job_directory, "cwloutput"),
                tmpdir=os.path.join(self._job_directory, "cwltmp"),
                stagedir=os.path.join(self._job_directory, "cwlstagedir"),
                use_container=False,
            ))
            self._is_command_line_job = hasattr(self._cwl_job, "command_line")

    def _normalize_job(self):
        # Somehow reuse whatever causes validate in cwltool... maybe?
        def pathToLoc(p):
            if "location" not in p and "path" in p:
                p["location"] = p["path"]
                del p["path"]
        process.fillInDefaults(self._tool_proxy._tool.tool["inputs"], self._input_dict)
        process.visit_class(self._input_dict, ("File", "Directory"), pathToLoc)
        # TODO: Why doesn't fillInDefault fill in locations instead of paths?
        process.normalizeFilesDirs(self._input_dict)
        # TODO: validate like cwltool process _init_job.
        #    validate.validate_ex(self.names.get_name("input_record_schema", ""), builder.job,
        #                         strict=False, logger=_logger_validation_warnings)

    def _select_resources(self, request):
        new_request = request.copy()
        new_request["cores"] = "$GALAXY_SLOTS"
        return new_request

    @property
    def command_line(self):
        if self.is_command_line_job:
            return self.cwl_job().command_line
        else:
            return ["true"]

    @property
    def stdin(self):
        if self.is_command_line_job:
            return self.cwl_job().stdin
        else:
            return None

    @property
    def stdout(self):
        if self.is_command_line_job:
            return self.cwl_job().stdout
        else:
            return None

    @property
    def environment(self):
        if self.is_command_line_job:
            return self.cwl_job().environment
        else:
            return {}

    @property
    def generate_files(self):
        if self.is_command_line_job:
            return self.cwl_job().generatefiles
        else:
            return {}

    def _output_callback(self, out, process_status):
        if process_status == "success":
            self._final_output = out
        else:
            self._ok = False

        log.info("Output are %s, status is %s" % (out, process_status))

    def collect_outputs(self, tool_working_directory):
        if not self.is_command_line_job:
            self.cwl_job().run(
            )
            return self._final_output
        else:
            return self.cwl_job().collect_outputs(tool_working_directory)

    def save_job(self):
        job_file = JobProxy._job_file(self._job_directory)
        job_objects = {
            # "tool_path": os.path.abspath(self._tool_proxy._tool_path),
            "tool_representation": self._tool_proxy.to_persistent_representation(),
            "job_inputs": self._input_dict,
            "output_dict": self._output_dict,
        }
        json.dump(job_objects, open(job_file, "w"))

    def _output_extra_files_dir(self, output_name):
        output_id = self.output_id(output_name)
        return os.path.join(self._job_directory, "dataset_%s_files" % output_id)

    def output_id(self, output_name):
        output_id = self._output_dict[output_name]["id"]
        return output_id

    def output_path(self, output_name):
        output_id = self._output_dict[output_name]["path"]
        return output_id

    def output_secondary_files_dir(self, output_name, create=False):
        extra_files_dir = self._output_extra_files_dir(output_name)
        secondary_files_dir = os.path.join(extra_files_dir, SECONDARY_FILES_EXTRA_PREFIX)
        if create and not os.path.exists(secondary_files_dir):
            safe_makedirs(secondary_files_dir)
        return secondary_files_dir

    def stage_files(self):
        cwl_job = self.cwl_job()
        if hasattr(cwl_job, "pathmapper"):
            process.stageFiles(self.cwl_job().pathmapper, os.symlink, ignoreWritable=True)
        # else: expression tools do not have a path mapper.

    @staticmethod
    def _job_file(job_directory):
        return os.path.join(job_directory, JOB_JSON_FILE)


class WorkflowProxy(object):

    def __init__(self, workflow, workflow_path):
        self._workflow = workflow
        self._workflow_path = workflow_path

    @property
    def cwl_id(self):
        return self._workflow.tool["id"]

    def tool_references(self):
        """Fetch tool source definitions for all referenced tools."""
        references = []
        for step in self.step_proxies():
            references.extend(step.tool_references())
        return references

    def tool_reference_proxies(self):
        return map(lambda tool_object: cwl_tool_object_to_proxy(tool_object), self.tool_references())

    def step_proxies(self):
        proxies = []
        num_input_steps = len(self._workflow.tool['inputs'])
        for i, step in enumerate(self._workflow.steps):
            proxies.append(StepProxy(self, step, i + num_input_steps))
        return proxies

    @property
    def runnables(self):
        runnables = []
        for step in self._workflow.steps:
            if "run" in step.tool:
                runnables.append(step.tool["run"])
        return runnables

    def cwl_ids_to_index(self, step_proxies):
        index = 0
        cwl_ids_to_index = {}
        for i, input_dict in enumerate(self._workflow.tool['inputs']):
            cwl_ids_to_index[input_dict["id"]] = index
            index += 1

        for step_proxy in step_proxies:
            cwl_ids_to_index[step_proxy.cwl_id] = index
            index += 1

        return cwl_ids_to_index

    def input_connections_by_step(self, step_proxies):
        cwl_ids_to_index = self.cwl_ids_to_index(step_proxies)

        input_connections_by_step = []
        for step_proxy in step_proxies:
            input_connections_step = {}
            cwl_inputs = step_proxy._step.tool["inputs"]
            for cwl_input in cwl_inputs:
                cwl_input_id = cwl_input["id"]
                cwl_source_id = cwl_input["source"]
                step_name, input_name = split_step_references(cwl_input_id, multiple=False)
                # Consider only allow multiple if MultipleInputFeatureRequirement is enabled
                for (output_step_name, output_name) in split_step_references(cwl_source_id):
                    output_step_id = self.cwl_id + "#" + output_step_name
                    if output_step_id not in cwl_ids_to_index:
                        template = "Output [%s] does not appear in ID-to-index map [%s]."
                        msg = template % (output_step_id, cwl_ids_to_index)
                        raise AssertionError(msg)

                    if input_name not in input_connections_step:
                        input_connections_step[input_name] = []

                    input_connections_step[input_name].append({
                        "id": cwl_ids_to_index[output_step_id],
                        "output_name": output_name,
                        "input_type": "dataset"
                    })

            input_connections_by_step.append(input_connections_step)

        return input_connections_by_step

    def to_dict(self):
        name = os.path.basename(self._workflow_path)
        steps = {}

        step_proxies = self.step_proxies()
        input_connections_by_step = self.input_connections_by_step(step_proxies)
        index = 0
        for i, input_dict in enumerate(self._workflow.tool['inputs']):
            steps[index] = self.cwl_input_to_galaxy_step(input_dict, i)
            index += 1

        for i, step_proxy in enumerate(step_proxies):
            input_connections = input_connections_by_step[i]
            steps[index] = step_proxy.to_dict(input_connections)
            print steps[index]
            index += 1

        return {
            'name': name,
            'steps': steps,
            'annotation': self.cwl_object_to_annotation(self._workflow.tool),
        }

    def jsonld_id_to_label(self, id):
        return id.rsplit("#", 1)[-1]

    def cwl_input_to_galaxy_step(self, input, i):
        input_type = input["type"]
        input_as_dict = {
            "id": i,
            "label": self.jsonld_id_to_label(input["id"]),
            "position": {"left": 0, "top": 0},
            "annotation": self.cwl_object_to_annotation(input),
            "input_connections": {},  # Should the Galaxy API really require this? - Seems to.
        }

        if input_type == "File":
            input_as_dict["type"] = "data_input"
        elif isinstance(input_type, dict) and input_type.get("type") == "array":
            input_as_dict["type"] = "data_collection_input"
            input_as_dict["collection_type"] = "list"
        else:
            raise NotImplementedError()

        return input_as_dict

    def cwl_object_to_annotation(self, cwl_obj):
        return cwl_obj.get("doc", None)


def split_step_references(step_references, multiple=True):
    """Split a CWL step input or output reference into step id and name."""
    # Trim off the workflow id part of the reference.
    step_references = listify(step_references)
    split_references = []

    for step_reference in step_references:
        assert "#" in step_reference
        cwl_workflow_id, step_reference = step_reference.split("#", 1)

        # Now just grab the step name and input/output name.
        assert "#" not in step_reference
        if "/" in step_reference:
            step_name, io_name = step_reference.split("/", 1)
        else:
            # Referencing an input, not a step.
            # In Galaxy workflows input steps have an implicit output named
            # "output" for consistency with tools - in cwl land
            # just the input name is referenced.
            step_name = step_reference
            io_name = "output"
        split_references.append((step_name, io_name))

    if multiple:
        return split_references
    else:
        assert len(split_references) == 1
        return split_references[0]


class StepProxy(object):

    def __init__(self, workflow_proxy, step, index):
        self._workflow_proxy = workflow_proxy
        self._step = step
        self._index = index

    @property
    def cwl_id(self):
        return self._step.id

    def tool_references(self):
        # Return a list so we can handle subworkflows recursively in the future.
        return [self._step.embedded_tool]

    def to_dict(self, input_connections):
        # We are to the point where we need a content id for this. We got
        # figure that out - short term we can load everything up as an
        # in-memory tool and reference by the JSONLD ID I think. So workflow
        # proxy should force the loading of a tool.
        tool_proxy = cwl_tool_object_to_proxy(self.tool_references()[0])
        tool_hash = build_tool_hash(tool_proxy.to_persistent_representation())

        # We need to stub out null entries for things getting replaced by
        # connections. This doesn't seem ideal - consider just making Galaxy
        # handle this.
        tool_state = {}
        for input_name in input_connections.keys():
            tool_state[input_name] = None

        return {
            "id": self._index,
            "tool_hash": tool_hash,
            "label": self._workflow_proxy.jsonld_id_to_label(self._step.id),
            "position": {"left": 0, "top": 0},
            "tool_state": tool_state,
            "type": "tool",  # TODO: dispatch on type obviously...
            "annotation": self._workflow_proxy.cwl_object_to_annotation(self._step.tool),
            "input_connections": input_connections,
        }


def remove_pickle_problems(obj):
    """doc_loader does not pickle correctly"""
    if hasattr(obj, "doc_loader"):
        obj.doc_loader = None
    if hasattr(obj, "embedded_tool"):
        obj.embedded_tool = remove_pickle_problems(obj.embedded_tool)
    if hasattr(obj, "steps"):
        obj.steps = [remove_pickle_problems(s) for s in obj.steps]
    return obj


@six.add_metaclass(ABCMeta)
class WorkflowToolReference(object):
    pass


class EmbeddedWorkflowToolReference(WorkflowToolReference):
    pass


class ExternalWorkflowToolReference(WorkflowToolReference):
    pass


def _outer_field_to_input_instance(field):
    field_type = field_to_field_type(field)  # Must be a list if in here?
    if not isinstance(field_type, list):
        field_type = [field_type]

    name, label, description = _field_metadata(field)

    case_name = "_cwl__type_"
    case_label = "Specify Parameter %s As" % label

    def value_input(type_description):
        value_name = "_cwl__value_"
        value_label = label
        value_description = description
        return InputInstance(
            value_name,
            value_label,
            value_description,
            input_type=type_description.galaxy_param_type,
            collection_type=type_description.collection_type,
        )

    select_options = []
    case_options = []
    type_descriptions = type_descriptions_for_field_types(field_type)
    for type_description in type_descriptions:
        select_options.append({"value": type_description.name, "label": type_description.label})
        input_instances = []
        if type_description.name == "null":
            print("better be false! %s" % type_description.uses_param)
        if type_description.uses_param():
            input_instances.append(value_input(type_description))
            print("ii %s" % [i.to_dict() for i in input_instances])
        case_options.append((type_description.name, input_instances))

    # If there is more than one way to represent this parameter - produce a conditional
    # requesting user to ask for what form they want to submit the data in, else just map
    # a simple Galaxy parameter.
    if len(case_options) > 1:
        case_input = SelectInputInstance(
            name=case_name,
            label=case_label,
            description=False,
            options=select_options,
        )

        return ConditionalInstance(name, case_input, case_options)
    else:
        only_type_description = type_descriptions[0]
        return InputInstance(
            name, label, description, input_type=only_type_description.galaxy_param_type, collection_type=only_type_description.collection_type
        )

    # Older array to repeat handling, now we are just representing arrays as
    # dataset collections - we should offer a blended approach in the future.
    # if field_type in simple_map_type_map.keys():
    #     input_type = simple_map_type_map[field_type]
    #     return {"input_type": input_type, "array": False}
    # elif field_type == "array":
    #     if isinstance(field["type"], dict):
    #         array_type = field["type"]["items"]
    #     else:
    #         array_type = field["items"]
    #     if array_type in simple_map_type_map.keys():
    #         input_type = simple_map_type_map[array_type]
    #     return {"input_type": input_type, "array": True}
    # else:
    #     raise Exception("Unhandled simple field type encountered - [%s]." % field_type)


def _field_metadata(field):
    name = field["name"]
    label = field.get("label", None)
    description = field.get("doc", None)
    return name, label, description


def _simple_field_to_output(field):
    name = field["name"]
    output_data_class = field["type"]
    output_instance = OutputInstance(
        name,
        output_data_type=output_data_class,
        output_type=OUTPUT_TYPE.GLOB
    )
    return output_instance


class ConditionalInstance(object):

    def __init__(self, name, case, whens):
        self.input_type = INPUT_TYPE.CONDITIONAL
        self.name = name
        self.case = case
        self.whens = whens

    def to_dict(self):

        as_dict = dict(
            name=self.name,
            type=INPUT_TYPE.CONDITIONAL,
            test=self.case.to_dict(),
            when=odict(),
        )
        for value, block in self.whens:
            as_dict["when"][value] = [i.to_dict() for i in block]

        return as_dict


class SelectInputInstance(object):

    def __init__(self, name, label, description, options):
        self.input_type = INPUT_TYPE.SELECT
        self.name = name
        self.label = label
        self.description = description
        self.options = options

    def to_dict(self):
        # TODO: serialize options...
        as_dict = dict(
            name=self.name,
            label=self.label or self.name,
            help=self.description,
            type=self.input_type,
            options=self.options,
        )
        return as_dict


class InputInstance(object):

    def __init__(self, name, label, description, input_type, array=False, area=False, collection_type=None):
        self.input_type = input_type
        self.collection_type = collection_type
        self.name = name
        self.label = label
        self.description = description
        self.required = True
        self.array = array
        self.area = area

    def to_dict(self, itemwise=True):
        if itemwise and self.array:
            as_dict = dict(
                type="repeat",
                name="%s_repeat" % self.name,
                title="%s" % self.name,
                blocks=[
                    self.to_dict(itemwise=False)
                ]
            )
        else:
            as_dict = dict(
                name=self.name,
                label=self.label or self.name,
                help=self.description,
                type=self.input_type,
                optional=not self.required,
            )
            if self.area:
                as_dict["area"] = True

            if self.input_type == INPUT_TYPE.INTEGER:
                as_dict["value"] = "0"
            if self.input_type == INPUT_TYPE.FLOAT:
                as_dict["value"] = "0.0"
            elif self.input_type == INPUT_TYPE.DATA_COLLECTON:
                as_dict["collection_type"] = "record"

        return as_dict


OUTPUT_TYPE = Bunch(
    GLOB="glob",
    STDOUT="stdout",
)


# TODO: Different subclasses - this is representing different types of things.
class OutputInstance(object):

    def __init__(self, name, output_data_type, output_type, path=None, fields=None):
        self.name = name
        self.output_data_type = output_data_type
        self.output_type = output_type
        self.path = path
        self.fields = fields


__all__ = (
    'tool_proxy',
    'load_job_proxy',
)
