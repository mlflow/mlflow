import mlflow
import pytest
from mlflow.utils.autologging_utils import try_mlflow_log, get_unspecified_default_args, \
    log_fn_args_as_params


@pytest.mark.large
def test_get_unspecified_default_args():

    # Example function signature we are testing on
    def fn(arg1, default1=1, default2=2):
        pass

    no_default_dict = get_unspecified_default_args(['arg1', 'default1'], {'default2': 42},
                                                   ['arg1', 'default1', 'default2'], [1, 2])

    assert len(no_default_dict.keys()) == 0

    no_default_dict2 = get_unspecified_default_args(['arg1', 'default1', 'default2'], {},
                                                    ['arg1', 'default1', 'default2'], [1, 2])

    assert len(no_default_dict2.keys()) == 0

    no_default_dict3 = get_unspecified_default_args(['arg1'], {'default1': 42, 'default2': 42},
                                                    ['arg1', 'default1', 'default2'], [1, 2])

    assert len(no_default_dict3.keys()) == 0

    no_default_dict4 = get_unspecified_default_args([], {'arg1': 42, 'default1': 42, 'default2': 42},
                                                    ['arg1', 'default1', 'default2'], [1, 2])

    assert len(no_default_dict4.keys()) == 0

    one_default_dict = get_unspecified_default_args(['user_arg'], {'default1': 42},
                                                    ['arg1', 'default1', 'default2'], [1, 2])

    assert len(one_default_dict.keys()) == 1
    assert 'default2' in one_default_dict
    assert one_default_dict['default2'] == 2

    one_default_dict2 = get_unspecified_default_args(['user_arg'], {'default2': 42},
                                                     ['arg1', 'default1', 'default2'], [1, 2])

    assert len(one_default_dict2.keys()) == 1
    assert 'default1' in one_default_dict2
    assert one_default_dict2['default1'] == 1

    one_default_dict3 = get_unspecified_default_args([], {'arg1': 42, 'default1': 42},
                                                     ['arg1', 'default1', 'default2'], [1, 2])

    assert len(one_default_dict3.keys()) == 1
    assert 'default2' in one_default_dict3
    assert one_default_dict3['default2'] == 2

    one_default_dict4 = get_unspecified_default_args(['arg1', 'default1'], {},
                                                     ['arg1', 'default1', 'default2'], [1, 2])

    assert len(one_default_dict4.keys()) == 1
    assert 'default2' in one_default_dict4
    assert one_default_dict4['default2'] == 2

    two_default_dict = get_unspecified_default_args(['arg1'], {},
                                                    ['arg1', 'default1', 'default2'], [1, 2])

    assert len(two_default_dict.keys()) == 2
    assert 'default1' in two_default_dict
    assert two_default_dict['default1'] == 1
    assert 'default2' in two_default_dict
    assert two_default_dict['default2'] == 2

    two_default_dict2 = get_unspecified_default_args([], {'arg1': 42},
                                                     ['arg1', 'default1', 'default2'], [1, 2])

    assert len(two_default_dict2.keys()) == 2
    assert 'default1' in two_default_dict2
    assert two_default_dict2['default1'] == 1
    assert 'default2' in two_default_dict2
    assert two_default_dict2['default2'] == 2

    # Test function signature for the following tests
    def fn_only_default(default1=1, default2=2, default3=3):
        pass

    three_default_dict = get_unspecified_default_args([], {},
                                                      ['default1', 'default2', 'default3'], [1, 2, 3])

    assert len(three_default_dict.keys()) == 3
    assert 'default1' in three_default_dict
    assert three_default_dict['default1'] == 1
    assert 'default2' in three_default_dict
    assert three_default_dict['default2'] == 2
    assert 'default3' in three_default_dict
    assert three_default_dict['default3'] == 3

    middle_default_dict = get_unspecified_default_args([], {'default2': 42},
                                                       ['default1', 'default2', 'default3'], [1, 2, 3])

    assert len(middle_default_dict.keys()) == 2
    assert 'default1' in middle_default_dict
    assert middle_default_dict['default1'] == 1
    assert 'default3' in middle_default_dict
    assert three_default_dict['default3'] == 3


@pytest.fixture
def start_run():
    mlflow.start_run()
    yield
    mlflow.end_run()


def dummy_fn(arg1, arg2='value2', arg3='value3'):
    pass


test_args = [([], {'arg1': 'value_x', 'arg2': 'value_y'}, ['value_x', 'value_y', 'value3']),
             (['value_x'], {'arg2': 'value_y'}, ['value_x', 'value_y', 'value3']),
             (['value_x'], {'arg3': 'value_z'}, ['value_x', 'value2', 'value_z']),
             (['value_x', 'value_y'], {}, ['value_x', 'value_y', 'value3']),
             (['value_x', 'value_y', 'value_z'], {}, ['value_x', 'value_y', 'value_z']),
             ([], {'arg1': 'value_x', 'arg2': 'value_y', 'arg3': 'value_z'}, ['value_x', 'value_y', 'value_z'])]


@pytest.mark.large
@pytest.mark.parametrize('args,kwargs,expected', test_args)
def test_log_fn_args_as_params(args, kwargs, expected, start_run):
    # can log defaults
    # can log args
    # can log kwargs
    print("FIXTURE RUN: " + str(mlflow.active_run().info.run_id))
    log_fn_args_as_params(dummy_fn, args, kwargs)
    params = mlflow.active_run().data.params
    for arg, value in zip(['arg1', 'arg2', 'arg3'], expected):
        print(params)
        assert arg in params
        assert params[arg] == value

    # can log all args
    # can log all kwargs


# ignores unlogged
@pytest.mark.large
def test_log_fn_args_as_params_ignores_unlogged(start_run):
    params = ('arg1', {'arg2': 'value'}, ['arg1', 'arg2', 'arg3'])

