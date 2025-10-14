from clint.rules.base import Rule
from clint.rules.do_not_disable import DoNotDisable
from clint.rules.docstring_param_order import DocstringParamOrder
from clint.rules.empty_notebook_cell import EmptyNotebookCell
from clint.rules.example_syntax_error import ExampleSyntaxError
from clint.rules.extraneous_docstring_param import ExtraneousDocstringParam
from clint.rules.forbidden_deprecation_warning import ForbiddenDeprecationWarning
from clint.rules.forbidden_set_active_model_usage import ForbiddenSetActiveModelUsage
from clint.rules.forbidden_top_level_import import ForbiddenTopLevelImport
from clint.rules.forbidden_trace_ui_in_notebook import ForbiddenTraceUIInNotebook
from clint.rules.get_artifact_uri import GetArtifactUri
from clint.rules.implicit_optional import ImplicitOptional
from clint.rules.incorrect_type_annotation import IncorrectTypeAnnotation
from clint.rules.invalid_abstract_method import InvalidAbstractMethod
from clint.rules.invalid_experimental_decorator import InvalidExperimentalDecorator
from clint.rules.isinstance_union_syntax import IsinstanceUnionSyntax
from clint.rules.lazy_builtin_import import LazyBuiltinImport
from clint.rules.lazy_module import LazyModule
from clint.rules.log_model_artifact_path import LogModelArtifactPath
from clint.rules.markdown_link import MarkdownLink
from clint.rules.missing_docstring_param import MissingDocstringParam
from clint.rules.missing_notebook_h1_header import MissingNotebookH1Header
from clint.rules.mlflow_class_name import MlflowClassName
from clint.rules.mock_patch_as_decorator import MockPatchAsDecorator
from clint.rules.mock_patch_dict_environ import MockPatchDictEnviron
from clint.rules.multi_assign import MultiAssign
from clint.rules.nested_mock_patch import NestedMockPatch
from clint.rules.no_class_based_tests import NoClassBasedTests
from clint.rules.no_rst import NoRst
from clint.rules.no_shebang import NoShebang
from clint.rules.os_chdir_in_test import OsChdirInTest
from clint.rules.os_environ_delete_in_test import OsEnvironDeleteInTest
from clint.rules.os_environ_set_in_test import OsEnvironSetInTest
from clint.rules.pytest_mark_repeat import PytestMarkRepeat
from clint.rules.redundant_test_docstring import RedundantTestDocstring
from clint.rules.temp_dir_in_test import TempDirInTest
from clint.rules.test_name_typo import TestNameTypo
from clint.rules.thread_pool_executor_without_thread_name_prefix import (
    ThreadPoolExecutorWithoutThreadNamePrefix,
)
from clint.rules.typing_extensions import TypingExtensions
from clint.rules.unknown_mlflow_arguments import UnknownMlflowArguments
from clint.rules.unknown_mlflow_function import UnknownMlflowFunction
from clint.rules.unnamed_thread import UnnamedThread
from clint.rules.unparameterized_generic_type import UnparameterizedGenericType
from clint.rules.use_sys_executable import UseSysExecutable

ALL_RULES = {rule.name for rule in Rule.__subclasses__()}


__all__ = [
    "ALL_RULES",
    "Rule",
    "DoNotDisable",
    "DocstringParamOrder",
    "EmptyNotebookCell",
    "ExampleSyntaxError",
    "ExtraneousDocstringParam",
    "ForbiddenDeprecationWarning",
    "GetArtifactUri",
    "ForbiddenSetActiveModelUsage",
    "ForbiddenTopLevelImport",
    "ForbiddenTraceUIInNotebook",
    "ImplicitOptional",
    "IncorrectTypeAnnotation",
    "IsinstanceUnionSyntax",
    "InvalidAbstractMethod",
    "InvalidExperimentalDecorator",
    "LazyBuiltinImport",
    "LazyModule",
    "LogModelArtifactPath",
    "MarkdownLink",
    "MissingDocstringParam",
    "MissingNotebookH1Header",
    "MlflowClassName",
    "MockPatchDictEnviron",
    "MockPatchAsDecorator",
    "NestedMockPatch",
    "NoClassBasedTests",
    "NoRst",
    "NoShebang",
    "OsChdirInTest",
    "OsEnvironDeleteInTest",
    "OsEnvironSetInTest",
    "PytestMarkRepeat",
    "RedundantTestDocstring",
    "TempDirInTest",
    "TestNameTypo",
    "ThreadPoolExecutorWithoutThreadNamePrefix",
    "TypingExtensions",
    "UnknownMlflowArguments",
    "UnknownMlflowFunction",
    "MultiAssign",
    "UnnamedThread",
    "UnparameterizedGenericType",
    "UseSysExecutable",
]
