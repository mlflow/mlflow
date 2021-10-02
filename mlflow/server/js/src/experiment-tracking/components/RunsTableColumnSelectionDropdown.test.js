import React from 'react';
import { RunsTableColumnSelectionDropdown } from './RunsTableColumnSelectionDropdown';
import { SearchTree } from '../../common/components/SearchTree';
import { Metric, Param, RunInfo, RunTag } from '../sdk/MlflowMessages';
import Utils from '../../common/utils/Utils';
import { COLUMN_TYPES, ATTRIBUTE_COLUMN_LABELS } from '../constants';
import { mountWithIntl, shallowWithIntl } from '../../common/utils/TestUtils';

describe('RunsTableColumnSelectionDropdown', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      paramKeyList: [],
      metricKeyList: [],
      visibleTagKeyList: [],
      onCheck: jest.fn(),
      categorizedUncheckedKeys: {
        [COLUMN_TYPES.ATTRIBUTES]: [],
        [COLUMN_TYPES.PARAMS]: [],
        [COLUMN_TYPES.METRICS]: [],
        [COLUMN_TYPES.TAGS]: [],
      },
    };

    commonProps = {
      ...minimalProps,
      paramKeyList: ['p1', 'p2'],
      metricKeyList: ['m1', 'm2'],
      visibleTagKeyList: ['t1', 't2'],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallowWithIntl(<RunsTableColumnSelectionDropdown {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render SearchTree with correct tree data', () => {
    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...commonProps} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('data')).toEqual([
      { key: 'attributes-Start Time', title: 'Start Time' },
      { key: 'attributes-Duration', title: 'Duration' },
      { key: 'attributes-User', title: 'User' },
      { key: 'attributes-Run Name', title: 'Run Name' },
      { key: 'attributes-Source', title: 'Source' },
      { key: 'attributes-Version', title: 'Version' },
      { key: 'attributes-Models', title: 'Models' },
      {
        title: 'Parameters',
        key: 'params',
        children: [
          { key: 'params-p1', title: 'p1' },
          { key: 'params-p2', title: 'p2' },
        ],
      },
      {
        title: 'Metrics',
        key: 'metrics',
        children: [
          { key: 'metrics-m1', title: 'm1' },
          { key: 'metrics-m2', title: 'm2' },
        ],
      },
      {
        title: 'Tags',
        key: 'tags',
        children: [
          { key: 'tags-t1', title: 't1' },
          { key: 'tags-t2', title: 't2' },
        ],
      },
    ]);
  });

  test('should check all keys by default', () => {
    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...commonProps} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('checkedKeys')).toEqual([
      'attributes-Start Time',
      'attributes-Duration',
      'attributes-User',
      'attributes-Run Name',
      'attributes-Source',
      'attributes-Version',
      'attributes-Models',
      'params-p1',
      'params-p2',
      'metrics-m1',
      'metrics-m2',
      'tags-t1',
      'tags-t2',
    ]);
  });

  test('should not check keys marked as unchecked', () => {
    const props = {
      ...commonProps,
      categorizedUncheckedKeys: {
        [COLUMN_TYPES.ATTRIBUTES]: ['User', 'Run Name', 'Source', 'Models', 'Version'],
        [COLUMN_TYPES.PARAMS]: ['p1'],
        [COLUMN_TYPES.METRICS]: ['m1'],
        [COLUMN_TYPES.TAGS]: ['t1'],
      },
    };
    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...props} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('checkedKeys')).toEqual([
      'attributes-Start Time',
      'attributes-Duration',
      'params-p2',
      'metrics-m2',
      'tags-t2',
    ]);
  });

  test('handleDiffViewCheckboxChange changes state correctly', () => {
    const getCategorizedColumnsDiffViewSpy = jest.fn();

    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...commonProps} />);
    instance = wrapper.instance();
    instance.getCategorizedColumnsDiffView = getCategorizedColumnsDiffViewSpy;

    // Checkbox unmarked by default
    expect(wrapper.state().diffViewSelected).toBe(false);

    // Checkbox marked
    instance.handleDiffViewCheckboxChange();
    expect(wrapper.state().diffViewSelected).toBe(true);
    expect(getCategorizedColumnsDiffViewSpy).toHaveBeenCalledTimes(1);
    expect(instance.props.onCheck).toHaveBeenCalledTimes(1);

    // Checkbox unmarked
    instance.handleDiffViewCheckboxChange();
    expect(wrapper.state().diffViewSelected).toBe(false);
    expect(getCategorizedColumnsDiffViewSpy).toHaveBeenCalledTimes(1);
    expect(instance.props.onCheck).toHaveBeenCalledTimes(2);
    expect(instance.props.onCheck).toHaveBeenLastCalledWith({
      [COLUMN_TYPES.ATTRIBUTES]: [],
      [COLUMN_TYPES.PARAMS]: [],
      [COLUMN_TYPES.METRICS]: [],
      [COLUMN_TYPES.TAGS]: [],
    });
  });

  test('getCategorizedColumnsDiffView returns the correct column keys to uncheck', () => {
    const runInfos = [
      RunInfo.fromJs({
        run_uuid: 'run-id1',
        experiment_id: '3',
        status: 'FINISHED',
        start_time: 1,
        end_time: 1,
        artifact_uri: 'dummypath',
        lifecycle_stage: 'active',
      }),
      RunInfo.fromJs({
        run_uuid: 'run-id2',
        experiment_id: '3',
        status: 'FINISHED',
        start_time: 2,
        end_time: 2,
        artifact_uri: 'dummypath',
        lifecycle_stage: 'active',
      }),
    ];
    const paramKeyList = ['param1', 'param2', 'param3', 'param4'];
    const metricKeyList = ['metric1', 'metric2', 'metric3', 'metric4'];
    const visibleTagKeyList = ['tag1', 'tag2', 'tag3'];
    const paramsList = [
      [
        Param.fromJs({ key: 'param1', value: '1' }),
        Param.fromJs({ key: 'param2', value: '1' }),
        Param.fromJs({ key: 'param3', value: '1' }),
      ],
      [Param.fromJs({ key: 'param1', value: '1' }), Param.fromJs({ key: 'param2', value: '2' })],
    ];
    const metricsList = [
      [
        Metric.fromJs({ key: 'metric1', value: '1' }),
        Metric.fromJs({ key: 'metric2', value: '1' }),
        Metric.fromJs({ key: 'metric3', value: '1' }),
      ],
      [
        Metric.fromJs({ key: 'metric1', value: '1' }),
        Metric.fromJs({ key: 'metric2', value: '2' }),
      ],
    ];
    const tagsList = [
      {
        tag1: RunTag.fromJs({ key: 'tag1', value: '1' }),
        tag2: RunTag.fromJs({ key: 'tag2', value: '1' }),
        tag3: RunTag.fromJs({ key: 'tag3', value: '1' }),
        [Utils.runNameTag]: RunTag.fromJs({ key: [Utils.runNameTag], value: 'runname1' }),
        [Utils.userTag]: RunTag.fromJs({ key: [Utils.userTag], value: 'usertag1' }),
        [Utils.sourceNameTag]: RunTag.fromJs({ key: [Utils.sourceNameTag], value: 'sourcename1' }),
        [Utils.gitCommitTag]: RunTag.fromJs({ key: [Utils.gitCommitTag], value: 'gitcommit1' }),
      },
      {
        tag1: RunTag.fromJs({ key: 'tag1', value: '1' }),
        tag2: RunTag.fromJs({ key: 'tag2', value: '2' }),
        [Utils.runNameTag]: RunTag.fromJs({ key: [Utils.runNameTag], value: 'runname1' }),
        [Utils.userTag]: RunTag.fromJs({ key: [Utils.userTag], value: 'usertag2' }),
        [Utils.sourceNameTag]: RunTag.fromJs({ key: [Utils.sourceNameTag], value: 'sourcename1' }),
        [Utils.gitCommitTag]: RunTag.fromJs({ key: [Utils.gitCommitTag], value: 'gitcommit2' }),
      },
    ];
    const expectedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: [ATTRIBUTE_COLUMN_LABELS.RUN_NAME, ATTRIBUTE_COLUMN_LABELS.SOURCE],
      [COLUMN_TYPES.PARAMS]: ['param1', 'param4'],
      [COLUMN_TYPES.METRICS]: ['metric1', 'metric4'],
      [COLUMN_TYPES.TAGS]: ['tag1'],
    };
    const props = {
      ...minimalProps,
      runInfos,
      paramKeyList,
      metricKeyList,
      visibleTagKeyList,
      paramsList,
      metricsList,
      tagsList,
    };
    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...props} />);
    instance = wrapper.instance();
    expect(instance.getCategorizedColumnsDiffView()).toEqual(expectedUncheckedKeys);
  });
});
