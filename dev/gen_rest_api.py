"""
TODO: Clena up this file.
"""

import argparse
import json
import logging
import re
from enum import Enum
from textwrap import dedent

from texttable import Texttable

"""
The functions in here are simple functions to help in managing RST from python
"""


def gen_break():
    return "\n\n===========================\n\n"


def gen_h1(link_id, title):
    link = link_id
    return """

.. _{link}:

{name}
{name_header}

""".format(name=title, link=link, name_header="=" * len(title))


def gen_h2(link_id, title):
    link = link_id
    return """

.. _{link}:

{name}
{name_header}

""".format(name=title, link=link, name_header="-" * len(title))


def gen_page_title(title):
    link = title.lower().replace(" ", "-")
    return """
.. _{link}:

{name_header}
{name}
{name_header}

""".format(name=title, link=link, name_header="=" * len(title))


def ErrorHelper(msg):
    return "JSON Validation Error: {}".format(msg)


def validate_doc_public_json(docjson):
    logging.info("Validating doc_public.json file.")
    if "files" not in docjson:
        logging.error(docjson.keys())
        raise Exception(ErrorHelper("No 'files' key"))
    files = docjson["files"][0]
    logging.info("Checking 'content'")
    if "content" not in files:
        logging.error(files.keys())
        raise Exception(ErrorHelper("No 'content' key"))
    content = files["content"][0]
    logging.info("Checking 'message', 'service', 'enum'")
    if "message" not in content:
        logging.error(content.keys())
        raise Exception(ErrorHelper("No 'message' key"))
    if "service" not in content:
        logging.error(content.keys())
        raise Exception(ErrorHelper("No 'service' key"))
    if "enum" not in content:
        logging.error(content.keys())
        raise Exception(ErrorHelper("No 'enum' key"))
    logging.info("Structure Appears to be valid! Continuing...")


class MsgType(Enum):
    generic = 1
    request = 2
    response = 3


def gen_id(full_path):
    "Generates an ID from a proto path"
    return "".join(full_path)


class Field:
    """
    A Field is a sub part of a message. It declares the field names, types, and descriptions
    for a given message.
    """

    def __init__(self, full_path, name, description, field_type):
        logging.debug("Creating Field {}".format(name))
        self.id = gen_id(full_path)
        self.name = name
        self.description = description
        self.field_type = field_type

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def table_header():
        return ["Field Name", "Type", "Description"]

    def to_table(self):
        return [self.name, self.field_type, self.description]

    @staticmethod
    def _parse_name(field_details):
        declared_type = field_details["field_type"]
        if declared_type == "oneof":
            names = ["``{}``".format(x["field_name"]) for x in field_details["oneof"]]
            name = " OR ".join(names)
        else:
            name = field_details["field_name"]
        return name

    @staticmethod
    def _parse_type(field_details):
        """
        Parses the Field Type from a field object.
        as of 8/31/2016 this is either a 'oneof' or some generic type(int, string, etc)
        """
        declared_type = field_details["field_type"]
        if declared_type == "oneof":
            field_types = [Field._convert_to_link(x["field_type"]) for x in field_details["oneof"]]
            return " OR ".join(field_types)
        return Field._convert_to_link(field_details["field_type"])

    @staticmethod
    def _parse_description(field_details):
        """
        Parses the description from a field object.
        The description structure depends explicitly on the field type.
        """
        declared_type = field_details["field_type"]
        deprecated = "" if not field_details["deprecated"] else "\nThis field is deprecated.\n"
        required = "" if not field_details["validate_required"] else "\nThis field is required.\n"
        if declared_type == "oneof":
            options = []

            def to_lowercase_first_char(s):
                return s[:1].lower() + s[1:] if s else ""

            for name, obj in zip(
                Field._parse_name(field_details).split(" OR "), field_details["oneof"]
            ):
                options.append(
                    "\n\nIf {}, {}".format(name, to_lowercase_first_char(obj["description"]))
                )
            return "\n\n\n\n".join(options) + required + deprecated
        return field_details["description"] + required + deprecated

    @staticmethod
    def _convert_to_link(raw_string):
        "Optionally converts a raw string to an internal link"
        if "." in raw_string:
            return ":ref:`{}`".format(raw_string.replace(".", "").lower())
        return "``{}``".format(raw_string)

    @classmethod
    def parse_all_from(cls, field_list):
        """
        Parses all fields from a field list inside of a message.
        Returns an array of Field objects.
        """
        all_instances = []
        for field in field_list:
            full_path = field["full_path"]
            name = Field._parse_name(field)
            field_type = Field._parse_type(field)
            description = Field._parse_description(field)
            assert not field["deprecated"]
            if field["repeated"]:
                field_type = "An array of " + field_type
            vis = field["visibility"]
            if vis == "public":
                all_instances.append(cls(full_path, name, description, field_type))
        return all_instances


class Value:
    """
    A value is a sub part of an ProtoEnum. It declares the potential values
    that a given enum can take on.
    """

    def __init__(self, full_path, name, description):
        self.id = gen_id(full_path)
        self.name = name
        self.description = description

    def __repr__(self):
        return "{}".format(self.name)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def table_header():
        return ["Name", "Description"]

    def to_table(self):
        return [self.name, self.description]

    @classmethod
    def parse_all_from(cls, value_list):
        """
        Parses all values from a value list inside of an enumeration.
        Returns a list of Value Objects.
        """
        all_instances = []
        for value in value_list:
            all_instances.append(cls(value["full_path"], value["value"], value["description"]))
        return all_instances


class ProtoEnum:
    """
    A ProtoEnum is an Enum defined by the Proto json. It has a series of Values.
    """

    def __init__(self, full_path, name, description, values):
        self.id = gen_id(full_path)
        self.name = name
        self.description = description
        self.values = values

    def __repr__(self):
        return "{}\n {}".format(self.id, self.name)

    def __str__(self):
        return self.__repr__()

    def _generate_values_table(self):
        """
        Each Enum can contain 0 or more values. This generates a plain text table
        out of the value's information.
        Returns a string.
        """
        tbl = Texttable(max_width=200)
        header = Value.table_header()
        tbl.add_rows([header] + [f.to_table() for f in self.values])
        return tbl.draw()

    def to_rst(self):
        "Converts a ProtoEnum Instance into rst"
        values = self._generate_values_table()
        title = gen_h2(self.id, self.name)
        section = "\n{description}\n\n{values}".format(description=self.description, values=values)
        return title + section

    @classmethod
    def parse_all_from(cls, files):
        """
        Parses all ProtoEnums from the raw proto file json. (the top level `files` group).
        Returns an array of ProtoEnums
        """
        all_instances = []
        for proto_file in files:
            # Top-level enums
            enums = [x["enum"] for x in proto_file["content"] if x["enum"]]
            # Enums inside of messages
            for content in filter(lambda x: x["message"], proto_file["content"]):
                enums += content["message"].get("enums") or []
            for enum in enums:
                values = Value.parse_all_from(enum["values"])
                description = enum["description"]
                name = enum["name"]
                all_instances.append(cls(enum["full_path"], name, description, values))

        return all_instances


class Message:
    """
    Mirrors a Proto Message like those that we find inside of the proto definition files
    """

    def __init__(self, full_path, name, description, fields):
        logging.debug("Creating Message: {}".format(name))
        self.id = gen_id(full_path)
        self.name = name
        self.description = description
        self.fields = fields
        self.type = MsgType.generic

    def __repr__(self):
        return "{}".format(self.id)

    def __str__(self):
        return self.__repr__()

    def _generate_field_table(self):
        tbl = Texttable(max_width=200)
        header = [Field.table_header()]
        non_empty_fields = [f for f in self.fields if len(f.name) != 0]
        rows = [f.to_table() for f in non_empty_fields]
        tbl.add_rows(header + rows)
        return tbl.draw()

    def _generate_rst_title(self):
        if self.type == MsgType.request:
            return gen_h2(self.id, "Request Structure")
        elif self.type == MsgType.response:
            return gen_h2(self.id, "Response Structure")
        return gen_h2(self.id, self.name)

    def to_rst(self):
        if not self.fields:
            return ""

        fields = self._generate_field_table()
        title = self._generate_rst_title()

        section = "\n\n{description}\n\n\n{fields}".format(
            description=self.description, fields=fields
        )
        return title + section

    @classmethod
    def parse_all_from_list(cls, message_list):
        """
        Parses all messages from a list of messages along with their associated sub messages
        (recursively). Returns a list of messages
        """
        all_instances = []
        for msg in message_list:
            name = msg["name"]
            description = msg["description"]
            fields = Field.parse_all_from(msg["fields"])
            full_path = msg["full_path"]
            vis = msg["visibility"]
            if vis == "public":
                all_instances.append(cls(full_path, name, description, fields))
                if msg["messages"]:
                    sub_messages = Message.parse_all_from_list(msg["messages"])
                    all_instances.extend(sub_messages)

        return all_instances

    @classmethod
    def parse_all_from(cls, files):
        """
        Parses all messages from raw proto file list along with their associated sub messages
        (recursively). Returns a list of messages.
        """
        all_instances = []
        for proto_file in files:
            for content in filter(lambda x: x["message"], proto_file["content"]):
                message = content["message"]
                fields = Field.parse_all_from(message["fields"])
                description = message["description"]
                full_path = message["full_path"]
                name = message["name"]
                vis = message["visibility"]
                if vis == "public":
                    all_instances.append(cls(full_path, name, description, fields))
                    if message["messages"]:
                        sub_messages = Message.parse_all_from_list(message["messages"])
                        all_instances.extend(sub_messages)

        return all_instances


class Method:
    """
    A method belongs to a service and describes the methods that are
    available to that given service. A method also contains messages
    in the form of the request message and the response message.
    """

    def __init__(self, full_path, name, description, path, method, request, response, title):
        self.id = gen_id(full_path)
        self.name = name
        self.description = description
        self.path = path
        self.method = method
        self.request = gen_id(request)
        self.response = gen_id(response)
        self.request_message = None
        self.response_message = None
        self.API_VERSION = None
        self.title = title

    @classmethod
    def parse_all_from(cls, method_list):
        all_instances = []
        for method in method_list:
            fp = method["full_path"]
            n = method["name"]
            d = method["description"]
            p = method["rpc_options"]["path"]
            m = method["rpc_options"]["method"]
            req = method["request_full_path"]
            resp = method["response_full_path"]
            vis = method["rpc_options"]["visibility"]
            title = method["rpc_options"].get("rpc_doc_title", None)
            if vis == "public":
                all_instances.append(cls(fp, n, d, p, m, req, resp, title))

        return all_instances

    def __repr__(self):
        reqm = "NoMsg"
        if self.request_message:
            reqm = "HasMsg"

        resm = "NoMsg"
        if self.response_message:
            resm = "HasMsg"

        return "{}, {} ({}) -> {} ({})".format(
            self.name, "".join(self.request), reqm, "".join(self.response), resm
        )

    def __str__(self):
        return self.__repr__()

    def to_rst(self):
        if not self.API_VERSION:
            raise Exception("MUST SET API VERSION")
        prepped_title = self.title or " ".join(re.split(r"\W+", self.path)[2:]).title().lstrip()
        title = gen_h1(self.id, prepped_title)
        tbl = Texttable(max_width=200)
        tbl.add_rows(
            [
                ["Endpoint", "HTTP Method"],
                ["``{}{}``".format(self.API_VERSION, self.path), "``{}``".format(self.method)],
            ]
        )
        parameters = tbl.draw()
        body = """
{parameters}

{description}
""".format(parameters=parameters, description=self.description)
        ret_value = gen_break() + title + body + "\n\n"
        if self.request_message:
            ret_value += self.request_message.to_rst()
        if self.response_message:
            ret_value += self.response_message.to_rst()
        return ret_value


class Service:
    "A Service as defined in our Proto Files"

    def __init__(self, full_path, name, description, methods):
        self.id = gen_id(full_path)
        self.name = name
        self.description = description
        self.methods = methods

    @classmethod
    def parse_all_from(cls, files):
        """
        Parses all Services from a list of proto files as found in doc_public.json
        returns a list of services.
        """
        all_instances = []
        for proto_file in files:
            for content in filter(lambda x: x["service"], proto_file["content"]):
                service = content["service"]
                methods = Method.parse_all_from(service["methods"])
                name = service["name"]
                description = service["description"]
                all_instances.append(cls(service["full_path"], name, description, methods))
        return all_instances

    def __repr__(self):
        return "{}\n Methods: {}".format(self.name, "\n  ".join([str(m) for m in self.methods]))

    def __str__(self):
        return self.__repr__()

    def to_rst(self, method_order=None):
        sections = []
        sorted_methods = sorted(self.methods, key=lambda x: x.name)
        if method_order:
            method_map = dict(zip(method_order, range(0, len(method_order))))
            sorted_methods = sorted(self.methods, key=lambda x: method_map[x.request])
        for method in sorted_methods:
            sections.append(method.to_rst())
        return "".join(sections)


class API:
    """
    An API is the only user facing module inside of this library. We specify an API
    along with a set of related endpoints.
    """

    def __init__(self, name, description, api_version, dstPath, valid_proto_files):
        self.name = name
        self.description = description
        self.dstPath = dstPath
        self.valid_proto_files = valid_proto_files
        self.services = None
        self.messages = None
        self.enums = None
        self.comments = None
        self.API_VERSION = api_version
        self.file_filter = lambda x: x["filename"] in self.valid_proto_files

    def __repr__(self):
        return "{}".format(self.name)

    def __str__(self):
        return self.__repr__()

    def validate_messages(self, valid_messages):
        logging.info("Validating Messages for {}".format(self))
        valid = set(valid_messages)
        actual = {m.id for m in self.messages}
        diff = actual - valid
        if diff:
            logging.error(
                """
The Specified Messages {} are not specified as a valid message for the docs but
visibility is not set to PUBLIC_UNDOCUMENTED or PRIVATE.
Please add it docs/api_sphinx_build.py under the relevant API endpoint or set it
to the correct visibility level.
""".format(diff)
            )
            raise

    def set_services(self, proto_file_list):
        """Parses the relevant services from a generic proto file list"""
        logging.debug("starting service generation")
        logging.debug(f"Starts with total of: {len(proto_file_list)}")
        services_proto_list = list(filter(self.file_filter, proto_file_list))
        try:
            assert len(services_proto_list) != 0
            assert len(services_proto_list) == len(self.valid_proto_files), (
                len(services_proto_list),
                len(self.valid_proto_files),
            )
        except AssertionError:
            logging.error("length cannot be 0. This is likely due to a name error")
            for f in self.valid_proto_files:
                logging.error("Valid Proto File: {}".format(f))
            for f in services_proto_list:
                logging.error("Actual Proto File: {}".format(f))
            logging.error("Maybe someone changed the name/location of a proto file?")
            raise
        self.services = sorted(Service.parse_all_from(services_proto_list), key=lambda x: x.name)
        logging.debug("completed service generation")

    def set_messages(self, proto_file_list):
        """Parses the relevant messages from a generic proto file list"""
        logging.debug("starting message generation")
        logging.debug(f"Starts with total of: {len(proto_file_list)}")
        messages_proto_list = list(filter(self.file_filter, proto_file_list))
        try:
            assert len(messages_proto_list) != 0
            assert len(messages_proto_list) == len(self.valid_proto_files)
        except AssertionError:
            logging.error("length cannot be 0. This is likely due to a name error")
            for f in self.valid_proto_files:
                logging.error("Valid Proto File: {}".format(f))
            logging.error("Maybe someone changed the name/location of a proto file?")
            raise
            raise

        self.messages = sorted(Message.parse_all_from(messages_proto_list), key=lambda x: x.name)
        logging.debug("completed message generation")

    def set_enums(self, proto_file_list):
        """Parses the relevant enums from a generic proto file list"""
        logging.debug("starting enum generation")
        logging.debug(f"Starts with total of: {len(proto_file_list)}")
        enums_proto_list = list(filter(self.file_filter, proto_file_list))
        try:
            assert len(enums_proto_list) != 0
            assert len(enums_proto_list) == len(self.valid_proto_files)
        except AssertionError:
            logging.error("length cannot be 0. This is likely due to a name error")
            for f in self.valid_proto_files:
                logging.error("Valid Proto File: {}".format(f))
            logging.error("Maybe someone changed the name/location of a proto file?")
            raise
        self.enums = sorted(ProtoEnum.parse_all_from(enums_proto_list), key=lambda x: x.name)
        logging.debug("completed enum generation")

    def connect_methods_messages(self):
        """
        Every service has an associated list of methods. Each of those methods have a request
        message and a response message. This method connects those two and labels the messages as
        such as well as adding them to the relevant methods.
        """
        for service in self.services:
            for method in service.methods:
                request_set = False
                response_set = False
                method.API_VERSION = self.API_VERSION
                for message in self.messages:
                    if message.id == method.request and not request_set:
                        method.request_message = message
                        message.type = MsgType.request
                        request_set = True
                        logging.debug("Set Request Message for {}".format(str(method)))
                    elif message.id == method.response and not response_set:
                        method.response_message = message
                        response_set = True
                        message.type = MsgType.response
                        logging.debug("Set Response Message for {}".format(str(method)))

                if not request_set:
                    logging.warn("Request not set {} for {}".format(method, str(self)))

                if not response_set:
                    logging.warn("Response not set {} for {}".format(method, str(self)))

    def set_all(self, proto_file_list):
        "Wraps all the above set methods"
        logging.info("Setting Services for {}".format(self.name))
        self.set_services(proto_file_list)
        logging.info("Finished Setting Services for {}".format(self.name))
        logging.info("Setting Messages for {}".format(self.name))
        self.set_messages(proto_file_list)
        logging.info("Finished Setting Messages for {}".format(self.name))
        logging.info("Setting Enums for {}".format(self.name))
        self.set_enums(proto_file_list)
        logging.info("Finished Setting Enums for {}".format(self.name))
        logging.info("Connecting Messages -> Services for {}".format(self.name))
        self.connect_methods_messages()
        logging.info("Finished Connecting Messages -> Services under {} API".format(self.name))

    def write_rst(self, method_order=None):
        try:
            assert (
                self.services is not None and self.messages is not None and self.enums is not None
            )
        except AssertionError:
            logging.error("We haven't parsed anything. You're using the module incorrectly")
            raise
        try:
            assert len(self.services) != 0 and len(self.messages) != 0
        except AssertionError:
            logging.error("There was likely an error parsing the doc_public.json file.")
            logging.error(
                "Services: %i Messages: %i Enums: %i",
                len(self.services),
                len(self.messages),
                len(self.enums),
            )
            raise
        services = [s.to_rst() for s in self.services]
        enums = [s.to_rst() for s in self.enums]
        generic_messages = [s.to_rst() for s in self.messages if s.type == MsgType.generic]

        with open(self.dstPath, "w") as f:
            f.write(gen_page_title("{} API".format(self.name)))
            f.write(self.description)
            f.write("\n.. contents:: Table of Contents\n    :local:\n    :depth: 1")
            f.write("".join(services))
            f.write(gen_h1(self.name + "add", "Data Structures"))
            f.write("".join(generic_messages))
            f.write("".join(enums))


logging.basicConfig(format="%(levelname)s:%(lineno)d:%(message)s", level=logging.INFO)


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""
This module converts a doc_public.json file created from the API files into
.rst files that can be integrated directly into the documentation.
"""
    )
    parser.add_argument("src", type=str, help="Path the doc public json FILE")
    parser.add_argument("dst", type=str, help="Destination FOLDER for the rst files")
    return parser


if __name__ == "__main__":
    src = "mlflow/protos/protos.json"
    dst = "docs/api_reference/source/rest-api.rst"

    API_VERSION = "2.0"
    logging.info(
        "SETTING API VERSION TO: {} EDIT api-sphinx-build.py TO CHANGE THIS VERSION".format(
            API_VERSION
        )
    )
    logging.info("Reading Source: " + src)

    with open(src) as f:
        docjson = json.load(f)

    validate_doc_public_json(docjson)

    proto_files = docjson["files"]

    mlflow_description = dedent("""
    The MLflow REST API allows you to create, list, and get experiments and runs, and log
    parameters, metrics, and artifacts. The API is hosted under the ``/api`` route on the MLflow
    tracking server. For example, to search for experiments on a tracking server hosted at
    ``http://localhost:5000``, make a POST request to ``http://localhost:5000/api/2.0/mlflow/experiments/search``.

    .. important::
        The MLflow REST API requires content type ``application/json`` for all POST requests.
    """)

    mlflow_protos = [
        "service.proto",
        "model_registry.proto",
        "webhooks.proto",
    ]
    valid_mlflow_messages = [
        # APIs
        "mlflowCreateExperiment",
        "mlflowListExperiments",
        "mlflowSearchExperiments",
        "mlflowGetExperiment",
        "mlflowGetExperimentByName",
        "mlflowDeleteExperiment",
        "mlflowRestoreExperiment",
        "mlflowUpdateExperiment",
        "mlflowCreateRun",
        "mlflowDeleteRun",
        "mlflowDeleteRuns",
        "mlflowRestoreRun",
        "mlflowRestoreRuns",
        "mlflowGetRun",
        "mlflowLogMetric",
        "mlflowLogBatch",
        "mlflowLogModel",
        "mlflowLogInputs",
        "mlflowLogBatchResponse",
        "mlflowSetExperimentTag",
        "mlflowDeleteExperimentTag",
        "mlflowSetTag",
        "mlflowDeleteTag",
        "mlflowLogParam",
        "mlflowGetMetricHistory",
        "mlflowSearchRuns",
        "mlflowListArtifacts",
        "mlflowUpdateRun",
        "mlflowPurgeDeletedState",
        "mlflowCreateRegisteredModel",
        "mlflowGetRegisteredModel",
        "mlflowRenameRegisteredModel",
        "mlflowUpdateRegisteredModel",
        "mlflowDeleteRegisteredModel",
        "mlflowListRegisteredModels",
        "mlflowGetLatestVersions",
        "mlflowCreateModelVersion",
        "mlflowGetModelVersion",
        "mlflowUpdateModelVersion",
        "mlflowDeleteModelVersion",
        "mlflowSearchModelVersions",
        "mlflowGetModelVersionDownloadUri",
        "mlflowTransitionModelVersionStage",
        "mlflowSearchRegisteredModels",
        "mlflowSetRegisteredModelTag",
        "mlflowSetModelVersionTag",
        "mlflowDeleteRegisteredModelTag",
        "mlflowDeleteModelVersionTag",
        # Responses
        "mlflowCreateExperimentResponse",
        "mlflowListExperimentsResponse",
        "mlflowSearchExperimentsResponse",
        "mlflowGetExperimentResponse",
        "mlflowGetExperimentByNameResponse",
        "mlflowDeleteExperimentResponse",
        "mlflowRestoreExperimentResponse",
        "mlflowUpdateExperimentResponse",
        "mlflowCreateRunResponse",
        "mlflowDeleteRunResponse",
        "mlflowDeleteRunsResponse",
        "mlflowRestoreRunResponse",
        "mlflowRestoreRunsResponse",
        "mlflowGetRunResponse",
        "mlflowLogMetricResponse",
        "mlflowLogInputsResponse",
        "mlflowLogParamResponse",
        "mlflowSetExperimentTagResponse",
        "mlflowSetTagResponse",
        "mlflowDeleteTagResponse",
        "mlflowGetMetricHistoryResponse",
        "mlflowSearchRunsResponse",
        "mlflowListArtifactsResponse",
        "mlflowUpdateRunResponse",
        "mlflowPurgeDeletedStateResponse",
        "mlflowCreateRegisteredModelResponse",
        "mlflowGetRegisteredModelResponse",
        "mlflowRenameRegisteredModelResponse",
        "mlflowUpdateRegisteredModelResponse",
        "mlflowListRegisteredModelsResponse",
        "mlflowGetLatestVersionsResponse",
        "mlflowCreateModelVersionResponse",
        "mlflowUpdateModelVersionResponse",
        "mlflowGetModelVersionResponse",
        "mlflowSearchModelVersionsResponse",
        "mlflowGetModelVersionDownloadUriResponse",
        "mlflowTransitionModelVersionStageResponse",
        "mlflowSearchRegisteredModelsResponse",
        "mlflowSetRegisteredModelTagResponse",
        "mlflowSetModelVersionTagResponse",
        "mlflowDeleteRegisteredModelTagResponse",
        "mlflowDeleteModelVersionTagResponse",
        "mlflowGetModelVersionByAliasResponse",
        # Other messages
        "mlflowExperiment",
        "mlflowRun",
        "mlflowRunInfo",
        "mlflowRunTag",
        "mlflowExperimentTag",
        "mlflowRunData",
        "mlflowRunInputs",
        "mlflowMetric",
        "mlflowParam",
        "mlflowFileInfo",
        "mlflowRegisteredModel",
        "mlflowModelVersion",
        "mlflowModelVersionDetailed",
        "mlflowModelVersionStatus",
        "mlflowRegisteredModelTag",
        "mlflowModelVersionTag",
        "mlflowDeleteRegisteredModelAlias",
        "mlflowGetModelVersionByAlias",
        "mlflowRegisteredModelAlias",
        "mlflowSetRegisteredModelAlias",
        "mlflowDatasetInput",
        "mlflowDataset",
        "mlflowInputTag",
        "mlflowDeploymentJobConnection",
        "mlflowModelVersionDeploymentJobState",
        "mlflowModelMetric",
        "mlflowModelOutput",
        "mlflowModelInput",
        "mlflowRunOutputs",
        "mlflowModelParam",
    ]

    mlflowAPI = API(
        "REST",
        mlflow_description,
        API_VERSION,
        dst,
        mlflow_protos,
    )
    mlflowAPI.set_all(proto_files)
    # Sort the methods based on the order in `valid_mlflow_messages` list.
    mlflowAPI.write_rst(valid_mlflow_messages)
