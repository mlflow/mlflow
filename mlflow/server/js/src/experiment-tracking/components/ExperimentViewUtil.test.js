import React from 'react';
import { shallow, mount } from 'enzyme';
import ExperimentViewUtil, { TreeNode } from './ExperimentViewUtil';
import { getModelVersionPageRoute } from '../../model-registry/routes';
import { BrowserRouter } from 'react-router-dom';
import { SEARCH_MAX_RESULTS } from '../actions';
import {
  ATTRIBUTE_COLUMN_LABELS,
  COLUMN_TYPES,
  DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  DEFAULT_ORDER_BY_KEY,
  DEFAULT_ORDER_BY_ASC,
  DEFAULT_START_TIME,
  DEFAULT_SHOW_MULTI_COLUMNS,
  DEFAULT_DIFF_SWITCH_SELECTED,
  DEFAULT_LIFECYCLE_FILTER,
  DEFAULT_MODEL_VERSION_FILTER,
} from '../constants';
import { Metric, Param, RunTag, RunInfo } from '../sdk/MlflowMessages';
import Utils from '../../common/utils/Utils';

const BASE_PATH = '/experiments/17/s';
const EXPERIMENT_ID = '17';

const createCategorizedUncheckedKeys = (arr) => ({
  [COLUMN_TYPES.ATTRIBUTES]: arr,
  [COLUMN_TYPES.PARAMS]: arr,
  [COLUMN_TYPES.METRICS]: arr,
  [COLUMN_TYPES.TAGS]: arr,
});

describe('ExperimentViewUtil', () => {
  test('getCheckboxForRow should render', () => {
    const component = ExperimentViewUtil.getCheckboxForRow(true, () => {}, 'div');
    const wrapper = shallow(component);
    expect(wrapper.length).toBe(1);
  });

  test('getRunInfoCellsForRow returns a row containing userid, start time, and status', () => {
    const runInfo = {
      user_id: 'user1',
      start_time: new Date('2020-01-02').getTime(),
      status: 'FINISHED',
    };
    const runInfoCells = ExperimentViewUtil.getRunInfoCellsForRow(
      runInfo,
      {},
      false,
      'div',
      () => {},
      [],
    );
    const renderedCells = runInfoCells.map((c) => mount(<BrowserRouter>{c}</BrowserRouter>));
    expect(renderedCells[0].find('.run-table-container').filter({ title: 'FINISHED' }).length).toBe(
      1,
    );
    const allText = renderedCells.map((c) => c.text()).join();
    expect(allText).toContain('user1');
    // The start_time is localized, so it may be anywhere from -12 to +14 hours, based on the
    // client's timezone.
    expect(
      allText.includes('2020-01-01') ||
        allText.includes('2020-01-02') ||
        allText.includes('2020-01-03'),
    ).toBeTruthy();
  });

  test('clicking on getRunMetadataHeaderCells sorts column if column is sortable', () => {
    const mockSortFn = jest.fn();
    const headerComponents = ExperimentViewUtil.getRunMetadataHeaderCells(
      mockSortFn,
      'user_id',
      true,
      'div',
      [],
    );
    // We assume that headerComponent[1] is the 'start_time' header
    const startTimeHeader = shallow(headerComponents[1]);
    startTimeHeader.find('.sortable').simulate('click');
    expect(mockSortFn.mock.calls[0][0]).toEqual(expect.stringContaining('start_time'));
    expect(mockSortFn.mock.calls[0][1]).toBeFalsy();
  });

  test('clicking on getRunMetadataHeaderCells does nothing if column is not sortable', () => {
    const mockSortFn = jest.fn();
    const headerComponents = ExperimentViewUtil.getRunMetadataHeaderCells(
      mockSortFn,
      'user_id',
      true,
      'div',
      [],
    );
    // We assume that headerComponent[0] is the 'status' header
    const statusHeader = shallow(headerComponents[0]);
    statusHeader.find('.run-table-container').simulate('click');
    expect(mockSortFn.mock.calls.length).toEqual(0);
  });

  test('getRunMetadataHeaderCells excludes excludedCols', () => {
    const headerComponents = ExperimentViewUtil.getRunMetadataHeaderCells(
      () => {},
      'user_id',
      true,
      'div',
      [ATTRIBUTE_COLUMN_LABELS.DATE],
    );
    const headers = headerComponents.map((c) => shallow(c));
    headers.forEach((h) => {
      expect(h.text()).not.toContain(ATTRIBUTE_COLUMN_LABELS.DATE);
    });

    // As a sanity check, let's make sure the headers contain some other column
    const userHeaders = headers.filter((h) => h.text() === ATTRIBUTE_COLUMN_LABELS.USER);
    expect(userHeaders.length).toBe(1);
  });

  test('computeMetricRanges returns the correct min and max value for a metric', () => {
    const metrics = [
      { key: 'foo', value: 1 },
      { key: 'foo', value: 2 },
      { key: 'foo', value: 0 },
    ];
    const metricsByRun = [metrics];
    const ranges = ExperimentViewUtil.computeMetricRanges(metricsByRun);
    expect(ranges.foo.min).toBe(0);
    expect(ranges.foo.max).toBe(2);
  });

  test('disable loadMoreButton when numRunsFromLatestSearch is not null and less than SEARCH_MAX_RESULTS', () => {
    expect(
      ExperimentViewUtil.disableLoadMoreButton({
        numRunsFromLatestSearch: null,
      }),
    ).toBe(false);

    expect(
      ExperimentViewUtil.disableLoadMoreButton({
        numRunsFromLatestSearch: SEARCH_MAX_RESULTS - 1,
      }),
    ).toBe(true);

    expect(
      ExperimentViewUtil.disableLoadMoreButton({
        numRunsFromLatestSearch: SEARCH_MAX_RESULTS,
      }),
    ).toBe(false);
  });

  test('get linked model cell displays model name with a single model version', () => {
    const modelName = 'model1';
    const model_versions = [{ name: modelName, version: 2 }];
    const linkedModelDiv = shallow(ExperimentViewUtil.getLinkedModelCell(model_versions));
    expect(
      linkedModelDiv
        .find('.model-version-link')
        .at(0)
        .props().href,
    ).toContain(getModelVersionPageRoute(model_versions[0].name, model_versions[0].version));
  });

  test('should not nest children if nestChildren is false', () => {
    const runInfos = [{ run_uuid: 1 }, { run_uuid: 2 }];
    const tagsList = [
      {
        'mlflow.parentRunId': {
          value: 2,
        },
      },
      {},
    ];
    const runsExpanded = { 1: true, 2: true };

    expect(
      ExperimentViewUtil.getRowRenderMetadata({
        runInfos,
        tagsList,
        runsExpanded,
        nestChildren: true,
      }),
    ).toEqual([
      {
        childrenIds: [1],
        expanderOpen: true,
        hasExpander: true,
        idx: 1,
        isParent: true,
        runId: 2,
      },
      { hasExpander: false, idx: 0, isParent: false },
    ]);

    expect(
      ExperimentViewUtil.getRowRenderMetadata({
        runInfos,
        tagsList,
        runsExpanded,
        nestChildren: false,
      }),
    ).toEqual([
      { hasExpander: false, idx: 0, isParent: true, runId: 1 },
      { hasExpander: false, idx: 1, isParent: true, runId: 2 },
    ]);
  });

  test('TreeNode finds the correct root', () => {
    const root = new TreeNode('root');
    const child = new TreeNode('child');
    child.parent = root;
    const grandchild = new TreeNode('grandchild');
    grandchild.parent = child;

    expect(grandchild.findRoot().value).toBe('root');
  });

  test('TreeNode knows which node is the root', () => {
    const root = new TreeNode('root');
    const child = new TreeNode('child');
    child.parent = root;

    expect(root.isRoot()).toBeTruthy();
    expect(child.isRoot()).toBeFalsy();
  });

  test('TreeNode detects a cycle', () => {
    const root = new TreeNode('root');
    const child = new TreeNode('child');
    child.parent = root;
    const child2 = new TreeNode('child2');
    child2.parent = root;
    const grandchild = new TreeNode('grandchild');
    grandchild.parent = child2;
    root.parent = grandchild;

    expect(grandchild.isCycle()).toBeTruthy();
  });

  test('getCategorizedUncheckedKeysDiffView returns the correct column keys to uncheck standard case', () => {
    const categorizedUncheckedKeys = DEFAULT_CATEGORIZED_UNCHECKED_KEYS;
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

    const createTags = (tags) => {
      // Converts {key: value, ...} to {key: RunTag(key, value), ...}
      return Object.entries(tags).reduce(
        (acc, [key, value]) => ({ ...acc, [key]: RunTag.fromJs({ key, value }) }),
        {},
      );
    };
    const tagsList = [
      createTags({
        tag1: '1',
        tag2: '1',
        tag3: '1',
        [Utils.runNameTag]: 'runname1',
        [Utils.gitCommitTag]: 'gitcommit1',
      }),
      createTags({
        tag1: '1',
        tag2: '2',
        [Utils.runNameTag]: 'runname1',
        [Utils.gitCommitTag]: 'gitcommit2',
      }),
    ];
    const expectedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: [ATTRIBUTE_COLUMN_LABELS.MODELS],
      [COLUMN_TYPES.PARAMS]: ['param1', 'param4'],
      [COLUMN_TYPES.METRICS]: ['metric1', 'metric4'],
      [COLUMN_TYPES.TAGS]: ['tag1'],
    };

    expect(
      ExperimentViewUtil.getCategorizedUncheckedKeysDiffView({
        categorizedUncheckedKeys,
        runInfos,
        paramKeyList,
        metricKeyList,
        paramsList,
        metricsList,
        tagsList,
      }),
    ).toEqual(expectedUncheckedKeys);
  });

  test('getCategorizedUncheckedKeysDiffView with columns already unchecked', () => {
    const categorizedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: [ATTRIBUTE_COLUMN_LABELS.RUN_NAME],
      [COLUMN_TYPES.PARAMS]: ['param2'],
      [COLUMN_TYPES.METRICS]: ['metric2'],
      [COLUMN_TYPES.TAGS]: ['tag2'],
    };
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

    const createTags = (tags) => {
      // Converts {key: value, ...} to {key: RunTag(key, value), ...}
      return Object.entries(tags).reduce(
        (acc, [key, value]) => ({ ...acc, [key]: RunTag.fromJs({ key, value }) }),
        {},
      );
    };
    const tagsList = [
      createTags({
        tag1: '1',
        tag2: '1',
        tag3: '1',
        [Utils.runNameTag]: 'runname1',
        [Utils.gitCommitTag]: 'gitcommit2',
      }),
      createTags({
        tag1: '1',
        tag2: '2',
        [Utils.runNameTag]: 'runname1',
        [Utils.gitCommitTag]: 'gitcommit1',
      }),
    ];
    const expectedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: [ATTRIBUTE_COLUMN_LABELS.RUN_NAME, ATTRIBUTE_COLUMN_LABELS.MODELS],
      [COLUMN_TYPES.PARAMS]: ['param2', 'param1', 'param4'],
      [COLUMN_TYPES.METRICS]: ['metric2', 'metric1', 'metric4'],
      [COLUMN_TYPES.TAGS]: ['tag2', 'tag1'],
    };

    expect(
      ExperimentViewUtil.getCategorizedUncheckedKeysDiffView({
        categorizedUncheckedKeys,
        runInfos,
        paramKeyList,
        metricKeyList,
        paramsList,
        metricsList,
        tagsList,
      }),
    ).toEqual(expectedUncheckedKeys);
  });

  test('getRestoredCategorizedUncheckedKeys no state change during switch', () => {
    const preSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys([]);
    const postSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2', 'k3']);
    const currCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2', 'k3']);
    const expectedCategorizedUncheckedKeys = createCategorizedUncheckedKeys([]);
    expect(
      ExperimentViewUtil.getRestoredCategorizedUncheckedKeys({
        preSwitchCategorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys,
        currCategorizedUncheckedKeys,
      }),
    ).toEqual(expectedCategorizedUncheckedKeys);
  });

  test('getRestoredCategorizedUncheckedKeys column unselected during switch', () => {
    const preSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys([]);
    const postSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2']);
    const currCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2', 'k3']);
    const expectedCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k3']);
    expect(
      ExperimentViewUtil.getRestoredCategorizedUncheckedKeys({
        preSwitchCategorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys,
        currCategorizedUncheckedKeys,
      }),
    ).toEqual(expectedCategorizedUncheckedKeys);
  });

  test('getRestoredCategorizedUncheckedKeys column selected during switch', () => {
    const preSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2']);
    const postSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2', 'k3']);
    const currCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1']);
    const expectedCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1']);
    expect(
      ExperimentViewUtil.getRestoredCategorizedUncheckedKeys({
        preSwitchCategorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys,
        currCategorizedUncheckedKeys,
      }),
    ).toEqual(expectedCategorizedUncheckedKeys);
  });

  test('getRestoredCategorizedUncheckedKeys column selected & unselected during switch', () => {
    const preSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2']);
    const postSwitchCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k2']);
    const currCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k3']);
    const expectedCategorizedUncheckedKeys = createCategorizedUncheckedKeys(['k1', 'k3']);
    expect(
      ExperimentViewUtil.getRestoredCategorizedUncheckedKeys({
        preSwitchCategorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys,
        currCategorizedUncheckedKeys,
      }),
    ).toEqual(expectedCategorizedUncheckedKeys);
  });
});

function expectSearchState(historyEntry, searchQueryParams) {
  const expectedPrefix = BASE_PATH + '?';
  expect(historyEntry.startsWith(expectedPrefix)).toBe(true);
  const search = historyEntry.substring(expectedPrefix.length);
  expect(search).toEqual(searchQueryParams);
}

describe('updateUrlWithViewState', () => {
  let defaultParameters;
  let history;
  beforeEach(() => {
    history = {};
    history.location = {};
    history.location.pathname = BASE_PATH;
    history.location.search = '';
    history.push = jest.fn();
    defaultParameters = {};
    defaultParameters.experimentId = EXPERIMENT_ID;
    defaultParameters.history = history;
    defaultParameters.searchInput = '';
    defaultParameters.lifecycleFilter = DEFAULT_LIFECYCLE_FILTER;
    defaultParameters.modelVersionFilter = DEFAULT_MODEL_VERSION_FILTER;
    defaultParameters.orderByKey = DEFAULT_ORDER_BY_KEY;
    defaultParameters.orderByAsc = DEFAULT_ORDER_BY_ASC;
    defaultParameters.startTime = DEFAULT_START_TIME;
    defaultParameters.showMultiColumns = DEFAULT_SHOW_MULTI_COLUMNS;
    defaultParameters.diffSwitchSelected = DEFAULT_DIFF_SWITCH_SELECTED;
    defaultParameters.preSwitchCategorizedUncheckedKeys = DEFAULT_CATEGORIZED_UNCHECKED_KEYS;
    defaultParameters.postSwitchCategorizedUncheckedKeys = DEFAULT_CATEGORIZED_UNCHECKED_KEYS;
  });

  test('updateUrlWithViewState updates URL correctly with default params', () => {
    ExperimentViewUtil.updateUrlWithViewState({
      ...defaultParameters,
    });
    expectSearchState(
      history.push.mock.calls[0][0],
      'startTime=ALL&orderByKey=attributes.start_time&lifecycle=Active&modelVersion=All%20Runs',
    );
  });

  test('updateUrlWithViewState updates URL correctly with orderByAsc true', () => {
    ExperimentViewUtil.updateUrlWithViewState({
      ...defaultParameters,
      orderByAsc: true,
    });
    expectSearchState(
      history.push.mock.calls[0][0],
      'startTime=ALL&orderByKey=attributes.start_time&orderByAsc=true&lifecycle=Active&modelVersion=All%20Runs',
    );
  });

  test('updateUrlWithViewState updates URL correctly with lifecycle & model filter', () => {
    ExperimentViewUtil.updateUrlWithViewState({
      ...defaultParameters,
      lifecycleFilter: 'life',
      modelVersionFilter: 'model',
    });
    expectSearchState(
      history.push.mock.calls[0][0],
      'startTime=ALL&orderByKey=attributes.start_time&lifecycle=life&modelVersion=model',
    );
  });

  test('updateUrlWithViewState updates URL correctly with searchInput', () => {
    ExperimentViewUtil.updateUrlWithViewState({
      ...defaultParameters,
      searchInput: 'metrics.metric0 > 3',
    });
    expectSearchState(
      history.push.mock.calls[0][0],
      'search=metrics.metric0%20%3E%203&startTime=ALL&orderByKey=attributes.start_time&lifecycle=Active&modelVersion=All%20Runs',
    );
  });

  test('updateUrlWithViewState updates URL correctly with showMultiColumns false', () => {
    ExperimentViewUtil.updateUrlWithViewState({
      ...defaultParameters,
      showMultiColumns: false,
    });
    expectSearchState(
      history.push.mock.calls[0][0],
      'startTime=ALL&orderByKey=attributes.start_time&lifecycle=Active&modelVersion=All%20Runs&showMultiColumns=false',
    );
  });

  test('updateUrlWithViewState updates URL correctly with diffSwitchSelected true', () => {
    ExperimentViewUtil.updateUrlWithViewState({
      ...defaultParameters,
      diffSwitchSelected: true,
    });
    expectSearchState(
      history.push.mock.calls[0][0],
      'startTime=ALL&orderByKey=attributes.start_time&lifecycle=Active&modelVersion=All%20Runs&diffSwitchSelected=true',
    );

    const preSwitchCategorizedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: ['a1'],
      [COLUMN_TYPES.PARAMS]: ['p1'],
      [COLUMN_TYPES.METRICS]: ['m1'],
      [COLUMN_TYPES.TAGS]: ['t1'],
    };

    const postSwitchCategorizedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: ['a2'],
      [COLUMN_TYPES.PARAMS]: ['p2'],
      [COLUMN_TYPES.METRICS]: ['m2'],
      [COLUMN_TYPES.TAGS]: ['t2'],
    };

    ExperimentViewUtil.updateUrlWithViewState({
      ...defaultParameters,
      diffSwitchSelected: true,
      preSwitchCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys,
    });
    expectSearchState(
      history.push.mock.calls[1][0],
      'startTime=ALL&orderByKey=attributes.start_time&lifecycle=Active' +
        '&modelVersion=All%20Runs&diffSwitchSelected=true' +
        '&preSwitchCategorizedUncheckedKeys%5Battributes%5D%5B0%5D=a1' +
        '&preSwitchCategorizedUncheckedKeys%5Bparams%5D%5B0%5D=p1' +
        '&preSwitchCategorizedUncheckedKeys%5Bmetrics%5D%5B0%5D=m1' +
        '&preSwitchCategorizedUncheckedKeys%5Btags%5D%5B0%5D=t1' +
        '&postSwitchCategorizedUncheckedKeys%5Battributes%5D%5B0%5D=a2' +
        '&postSwitchCategorizedUncheckedKeys%5Bparams%5D%5B0%5D=p2' +
        '&postSwitchCategorizedUncheckedKeys%5Bmetrics%5D%5B0%5D=m2' +
        '&postSwitchCategorizedUncheckedKeys%5Btags%5D%5B0%5D=t2',
    );
  });
});
