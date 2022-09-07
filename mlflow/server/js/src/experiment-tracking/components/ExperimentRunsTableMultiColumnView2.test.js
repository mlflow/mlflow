import React from 'react';
import { shallow } from 'enzyme';
import {
  ExperimentRunsTableMultiColumnView2Impl,
  ModelsCellRenderer,
  TagCellRenderer,
} from './ExperimentRunsTableMultiColumnView2';
import { COLUMN_TYPES, ATTRIBUTE_COLUMN_LABELS } from '../constants';
import { MemoryRouter as Router } from 'react-router-dom';
import { RunInfo } from '../sdk/MlflowMessages';
import { mountWithIntl } from '../../common/utils/TestUtils';

/**
 * Let's create our own mocked version of agGrid wrapper
 * that is not dynamically imported, otherwise deeply coupled
 * tests below will fail.
 */
/* eslint-disable global-require */
jest.mock('../../common/components/ag-grid/AgGridLoader', () => ({
  MLFlowAgGridLoader: (props) => {
    const AgGridReactImpl = require('@ag-grid-community/react').AgGridReact;
    return (
      <AgGridReactImpl
        modules={[require('@ag-grid-community/client-side-row-model').ClientSideRowModelModule]}
        {...props}
      />
    );
  },
}));
/* eslint-enable global-require */

function getChildColumnNames(columnDefs, parentName) {
  return columnDefs
    .find((header) => header.headerName === parentName)
    .children.map((childHeader) => {
      return childHeader.headerName;
    });
}

describe('ExperimentRunsTableMultiColumnView2', () => {
  let wrapper;
  let minimalProps;
  let commonProps;
  let setColumnDefsSpy;
  const loggedModels = [
    {
      artifactPath: 'somePath',
      flavors: ['keras'],
      utcTimeCreated: new Date('2020-10-22').getTime() / 1000,
    },
  ];

  beforeEach(() => {
    minimalProps = {
      experiments: [],
      runInfos: [],
      paramsList: [],
      metricsList: [],
      paramKeyList: [],
      metricKeyList: [],
      visibleTagKeyList: [],
      tagsList: [],
      modelVersionsByRunUuid: {},
      onSelectionChange: jest.fn(),
      onExpand: jest.fn(),
      onSortBy: jest.fn(),
      runsSelected: {},
      runsExpanded: {},
      handleLoadMoreRuns: jest.fn(),
      loadingMore: false,
      isLoading: false,
      categorizedUncheckedKeys: {
        [COLUMN_TYPES.ATTRIBUTES]: [],
        [COLUMN_TYPES.PARAMS]: [],
        [COLUMN_TYPES.METRICS]: [],
        [COLUMN_TYPES.TAGS]: [],
      },
      designSystemThemeApi: { theme: { colors: {} } },
    };
    commonProps = {
      ...minimalProps,
      metricKeyList: ['metric1', 'metric2'],
      paramKeyList: ['param1', 'param2'],
      visibleTagKeyList: ['tag1', 'tag2'],
      categorizedUncheckedKeys: {
        [COLUMN_TYPES.ATTRIBUTES]: [ATTRIBUTE_COLUMN_LABELS.DATE, ATTRIBUTE_COLUMN_LABELS.DURATION],
      },
    };
    setColumnDefsSpy = jest.fn();
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ExperimentRunsTableMultiColumnView2Impl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should show placeholder in logged model cells when no logged models', () => {
    const props = {
      value: {
        runUuid: 'runUuid',
        experimentId: 'experimentId',
        registeredModels: [],
        loggedModels: [],
      },
    };
    const output = ModelsCellRenderer(props);
    expect(output).toEqual('-');
  });

  test('should render logged model link with correct href', () => {
    const props = {
      ...commonProps,
      runInfos: [
        RunInfo.fromJs({
          artifact_uri: 'artifact_uri',
          end_time: 1,
          experiment_id: '0',
          lifecycle_stage: 'active',
          run_uuid: '123',
          start_time: 0,
          status: 'FINISHED',
          user_id: 'user_id',
          getRunUuid: () => '123',
        }),
      ],
      metricsList: [[]],
      paramsList: [[]],
      tagsList: [
        {
          'mlflow.log-model.history': {
            key: 'mlflow.log-model.history',
            value: `[{
              "run_id": "123",
              "artifact_path": "model",
              "utc_time_created": "2022-01-01 00:00:00.000000"
            }]`,
          },
        },
      ],
    };
    wrapper = mountWithIntl(<ExperimentRunsTableMultiColumnView2Impl {...props} />);
    const loggedModelLink = wrapper.find('.logged-model-link');
    expect(loggedModelLink).toHaveLength(1);
    expect(loggedModelLink.prop('href')).toEqual('./#/experiments/0/runs/123/artifactPath/model');
  });

  test('should show only logged model link if no registered models', () => {
    const props = {
      value: {
        runUuid: 'runUuid',
        experimentId: 'experimentId',
        loggedModels: loggedModels,
        registeredModels: [],
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('keras');
    expect(wrapper.find('.logged-model-link')).toHaveLength(1);
    expect(wrapper.find('.registered-model-link')).toHaveLength(0);
    expect(wrapper.html()).not.toContain('1 more');
  });

  test('should show correct logged and registered model links', () => {
    const tester = (modelVersions, expectedPath, linkSelector) => {
      const props = {
        value: {
          runUuid: 'runUuid',
          experimentId: 'experimentId',
          loggedModels: loggedModels,
          registeredModels: modelVersions,
        },
      };
      let output = ModelsCellRenderer(props);
      wrapper = shallow(<Router>{output}</Router>);
      expect(wrapper.find(linkSelector)).toHaveLength(1);

      let href = wrapper.find(linkSelector).prop('href');

      // Assume we're not in an iframe
      expect(href).toEqual(`./#/${expectedPath}`);

      // Assume we're in an iframe
      window.isTestingIframe = true;
      output = ModelsCellRenderer(props);
      wrapper = shallow(<Router>{output}</Router>);
      href = wrapper.find(linkSelector).prop('href');
      expect(href.endsWith(expectedPath)).toBe(true);
      expect(href.indexOf('http')).toEqual(0);
      expect(href.indexOf('http')).toEqual(href.lastIndexOf('http'));
      window.isTestingIframe = false;
    };
    tester([], 'experiments/experimentId/runs/runUuid/artifactPath/somePath', '.logged-model-link');

    tester(
      [
        {
          name: 'someModel',
          source: 'dbfs/runUuid/artifacts/somePath',
          version: 2,
        },
      ],
      'models/someModel/versions/2',
      '.registered-model-link',
    );
  });

  test('should only show registered model link if registered model', () => {
    const props = {
      value: {
        runUuid: 'someUuid',
        experimentId: 'experimentId',
        loggedModels: loggedModels,
        registeredModels: [
          {
            name: 'someName',
            source: 'dbfs/someUuid/artifacts/somePath',
            version: 2,
          },
        ],
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('someName');
    expect(wrapper.find('img[data-test-id="logged-model-icon"]')).toHaveLength(0);
    expect(wrapper.find('img[data-test-id="registered-model-icon"]')).toHaveLength(1);
  });

  /**
   * A model version's source may be semantically equivalent to a run artifact path
   * but syntactically distinct; this occurs when there are redundant or trailing
   * slashes present in the version source or run artifact path. This test verifies that,
   * in these cases, associated registered model links are still displayed correctly for runs
   * containing model artifacts with semantically equivalent paths.
   */
  test('should show registered model link for semantically equivalent artifact paths', () => {
    const runLoggedModels = [
      {
        artifactPath: 'somePath////subdir///',
        flavors: ['keras'],
        utcTimeCreated: new Date('2020-10-22').getTime() / 1000,
      },
    ];

    const props = {
      value: {
        runUuid: 'someUuid',
        loggedModels: runLoggedModels,
        registeredModels: [
          {
            name: 'someName',
            source: 'dbfs:///someUuid/artifacts///somePath///subdir/',
            version: 2,
          },
        ],
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('someName');
    expect(wrapper.find('img[data-test-id="logged-model-icon"]')).toHaveLength(0);
    expect(wrapper.find('img[data-test-id="registered-model-icon"]')).toHaveLength(1);
  });

  test('should show 1 more if two logged models', () => {
    const runLoggedModels = [
      {
        artifactPath: 'somePath',
        flavors: ['keras'],
        utcTimeCreated: new Date('2020-10-22').getTime() / 1000,
      },
      {
        artifactPath: 'someOtherPath',
        flavors: ['keras'],
        utcTimeCreated: new Date('2020-10-22').getTime() / 1000,
      },
    ];
    const props = {
      value: {
        runUuid: 'runUuid',
        experimentId: 'experimentId',
        loggedModels: runLoggedModels,
        registeredModels: [],
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('1 more');
  });

  test('getColumnDefs should return only attribute columns that are not unchecked', () => {
    const expectedColumnNames = [
      undefined,
      ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
      ATTRIBUTE_COLUMN_LABELS.USER,
      ATTRIBUTE_COLUMN_LABELS.SOURCE,
      ATTRIBUTE_COLUMN_LABELS.VERSION,
      ATTRIBUTE_COLUMN_LABELS.MODELS,
      'Metrics',
      'Parameters',
      'Tags',
    ];
    const expectedMetricColumnNames = ['metric1', 'metric2'];
    const expectedParameterColumnNames = ['param1', 'param2'];
    const expectedTagColumnNames = ['tag1', 'tag2'];

    wrapper = shallow(<ExperimentRunsTableMultiColumnView2Impl {...commonProps} />);
    const instance = wrapper.instance();
    const columnNames = instance.state.columnDefs.map((column) => {
      return column.headerName;
    });
    const metricColumnNames = getChildColumnNames(instance.state.columnDefs, 'Metrics');
    const paramColumnNames = getChildColumnNames(instance.state.columnDefs, 'Parameters');
    const tagColumnNames = getChildColumnNames(instance.state.columnDefs, 'Tags');

    expect(columnNames).toEqual(expectedColumnNames);
    expect(metricColumnNames).toEqual(expectedMetricColumnNames);
    expect(paramColumnNames).toEqual(expectedParameterColumnNames);
    expect(tagColumnNames).toEqual(expectedTagColumnNames);
  });

  test('getColumnDefs should not no longer return columnDef after column uncheck', () => {
    const expectedColumnNames = [
      undefined,
      ATTRIBUTE_COLUMN_LABELS.USER,
      ATTRIBUTE_COLUMN_LABELS.SOURCE,
      ATTRIBUTE_COLUMN_LABELS.VERSION,
      ATTRIBUTE_COLUMN_LABELS.MODELS,
      'Metrics',
      'Parameters',
      'Tags',
    ];
    const expectedMetricColumnNames = ['metric1'];
    const expectedParameterColumnNames = ['param1'];
    const expectedTagColumnNames = ['tag1'];

    wrapper = shallow(<ExperimentRunsTableMultiColumnView2Impl {...commonProps} />);
    const instance = wrapper.instance();
    instance.setColumnDefs = setColumnDefsSpy;

    wrapper.setProps({
      metricKeyList: expectedMetricColumnNames,
      paramKeyList: expectedParameterColumnNames,
      visibleTagKeyList: expectedTagColumnNames,
      categorizedUncheckedKeys: {
        [COLUMN_TYPES.ATTRIBUTES]: [
          ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
          ATTRIBUTE_COLUMN_LABELS.DATE,
          ATTRIBUTE_COLUMN_LABELS.DURATION,
        ],
      },
    });

    const columnNames = instance.getColumnDefs().map((column) => {
      return column.headerName;
    });
    const metricColumnNames = getChildColumnNames(instance.getColumnDefs(), 'Metrics');
    const paramColumnNames = getChildColumnNames(instance.getColumnDefs(), 'Parameters');
    const tagColumnNames = getChildColumnNames(instance.getColumnDefs(), 'Tags');

    expect(columnNames).toEqual(expectedColumnNames);
    expect(metricColumnNames).toEqual(expectedMetricColumnNames);
    expect(paramColumnNames).toEqual(expectedParameterColumnNames);
    expect(tagColumnNames).toEqual(expectedTagColumnNames);
    expect(setColumnDefsSpy).toHaveBeenCalledTimes(1);
  });

  describe('handleRowSelected', () => {
    // This will create ag-grid-like row node entity with setSelected mock function
    const createGridNode = (data) => ({
      data,
      setSelected: jest.fn(),
    });

    let allNodes;
    let rowsComponentInstance;

    beforeEach(() => {
      // Set of "all" rows living in the virtual agGrid row set.
      // Rows 1 and 2 are children of row 0.
      // Row 5 is child of row 4.
      // Row 6 is child of row 5.
      allNodes = [
        /* 0 */ createGridNode({
          runInfo: { run_uuid: '0' },
          runDateInfo: {
            isParent: true,
            expanderOpen: true,
            childrenIds: ['1', '2'],
          },
        }),
        /* 1 */ createGridNode({ runInfo: { run_uuid: '1' } }),
        /* 2 */ createGridNode({ runInfo: { run_uuid: '2' } }),
        /* 3 */ createGridNode({ runInfo: { run_uuid: '3' } }),
        /* 4 */ createGridNode({
          runInfo: { run_uuid: '4' },
          runDateInfo: {
            isParent: true,
            expanderOpen: true,
            childrenIds: ['5'],
          },
        }),
        /* 5 */ createGridNode({
          runInfo: { run_uuid: '5' },
          runDateInfo: {
            isParent: true,
            expanderOpen: true,
            childrenIds: ['6'],
          },
        }),
        /* 6 */ createGridNode({ runInfo: { run_uuid: '6' } }),
        /* 7 */ createGridNode({ runInfo: { run_uuid: '7' } }),
      ];

      const rowsWrapper = shallow(<ExperimentRunsTableMultiColumnView2Impl {...commonProps} />);
      rowsComponentInstance = rowsWrapper.instance();
      rowsComponentInstance.prevSelectRunUuids = [];
    });

    test('should select children only if the selection has changed', () => {
      // Initiate mock with selected rows count equal to previously selected rows (0)
      const eventData = {
        api: { getSelectedRows: () => [] },
        node: {
          isSelected: jest.fn(),
        },
      };

      // Perform the action
      rowsComponentInstance.handleRowSelected(eventData);

      // Assert that the node has not been even checked for selection
      expect(eventData.node.isSelected).not.toBeCalled();

      // Initiate mock with selected rows count different than previously selected rows
      const selectedEventData = {
        api: { getSelectedRows: () => [{}, {}] },
        node: {
          isSelected: jest.fn(),
        },
        data: {},
      };

      // Perform the action
      rowsComponentInstance.handleRowSelected(selectedEventData);

      // Assert that the node has been even checked for selection
      expect(selectedEventData.node.isSelected).toBeCalledTimes(1);
    });
    test('should properly calculate children marked for selection', () => {
      // Initiate mock with newly selected row #0
      rowsComponentInstance.handleRowSelected({
        api: {
          getSelectedRows: () => [{}, {}],
          forEachNode: (fn) => allNodes.forEach(fn),
        },
        node: {
          isSelected: () => true,
        },
        data: allNodes[0].data,
      });

      // Expect only children to be selected
      expect(allNodes[0].setSelected).not.toBeCalled();
      expect(allNodes[1].setSelected).toHaveBeenLastCalledWith(true, false, true);
      expect(allNodes[2].setSelected).toHaveBeenLastCalledWith(true, false, true);
      expect(allNodes[3].setSelected).not.toBeCalled();
    });
    test('should ensure that nested children are marked for selection', () => {
      // Initiate mock with newly selected row #4
      rowsComponentInstance.handleRowSelected({
        api: {
          getSelectedRows: () => [{}, {}],
          forEachNode: (fn) => allNodes.forEach(fn),
        },
        node: {
          isSelected: () => true,
        },
        data: allNodes[4].data,
      });

      // Expect only children to be selected
      expect(allNodes[4].setSelected).not.toBeCalled();
      expect(allNodes[5].setSelected).toHaveBeenLastCalledWith(true, false, true);
      expect(allNodes[6].setSelected).toHaveBeenLastCalledWith(true, false, true);
      expect(allNodes[7].setSelected).not.toBeCalled();
    });
    test('should unselect a subtree inside the selected tree', () => {
      const api = {
        getSelectedRows: () => [{}, {}],
        forEachNode: (fn) => allNodes.forEach(fn),
      };

      // First, select row #4
      rowsComponentInstance.handleRowSelected({
        api,
        node: {
          isSelected: () => true,
        },
        data: allNodes[4].data,
      });

      // Then deselect row #5
      rowsComponentInstance.handleRowSelected({
        api,
        node: {
          isSelected: () => false,
        },
        data: allNodes[5].data,
      });

      // We should have one call for row #6 - first selecting, then deselecting
      expect(allNodes[6].setSelected.mock.calls[0]).toEqual([true, false, true]);
      expect(allNodes[6].setSelected.mock.calls[1]).toEqual([false, false, true]);
      // Also we should have only one call for row #5 that selects it
      expect(allNodes[5].setSelected.mock.calls).toEqual([[true, false, true]]);
    });
  });

  test('getColumnDefs should return columnDef after column check', () => {
    const expectedColumnNames = [
      undefined,
      ATTRIBUTE_COLUMN_LABELS.DATE,
      ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
      ATTRIBUTE_COLUMN_LABELS.USER,
      ATTRIBUTE_COLUMN_LABELS.SOURCE,
      ATTRIBUTE_COLUMN_LABELS.VERSION,
      ATTRIBUTE_COLUMN_LABELS.MODELS,
      'Metrics',
      'Parameters',
      'Tags',
    ];
    const expectedMetricColumnNames = ['metric1', 'metric2', 'metric3'];
    const expectedParameterColumnNames = ['param1', 'param2', 'param3'];
    const expectedTagColumnNames = ['tag1', 'tag2', 'tag3'];

    wrapper = shallow(<ExperimentRunsTableMultiColumnView2Impl {...commonProps} />);
    const instance = wrapper.instance();
    instance.setColumnDefs = setColumnDefsSpy;

    wrapper.setProps({
      metricKeyList: expectedMetricColumnNames,
      paramKeyList: expectedParameterColumnNames,
      visibleTagKeyList: expectedTagColumnNames,
      categorizedUncheckedKeys: {
        [COLUMN_TYPES.ATTRIBUTES]: [ATTRIBUTE_COLUMN_LABELS.DURATION],
      },
    });

    const columnNames = instance.getColumnDefs().map((column) => {
      return column.headerName;
    });
    const metricColumnNames = getChildColumnNames(instance.getColumnDefs(), 'Metrics');
    const paramColumnNames = getChildColumnNames(instance.getColumnDefs(), 'Parameters');
    const tagColumnNames = getChildColumnNames(instance.getColumnDefs(), 'Tags');

    expect(columnNames).toEqual(expectedColumnNames);
    expect(metricColumnNames).toEqual(expectedMetricColumnNames);
    expect(paramColumnNames).toEqual(expectedParameterColumnNames);
    expect(tagColumnNames).toEqual(expectedTagColumnNames);
    expect(setColumnDefsSpy).toHaveBeenCalledTimes(1);
  });

  test('should render tag that is a valid http(s) as clickable link', () => {
    expect(TagCellRenderer({ value: 'https://some_url.com' })).toEqual(
      <a href='https://some_url.com' target='_blank' rel='noreferrer'>
        https://some_url.com
      </a>,
    );
    expect(TagCellRenderer({ value: 'http://some_url.com' })).toEqual(
      <a href='http://some_url.com' target='_blank' rel='noreferrer'>
        http://some_url.com
      </a>,
    );
    expect(TagCellRenderer({ value: 'some text' })).toEqual('some text');
    expect(TagCellRenderer({ value: 'some text https://some_url.com' })).toEqual(
      'some text https://some_url.com',
    );
  });
});
