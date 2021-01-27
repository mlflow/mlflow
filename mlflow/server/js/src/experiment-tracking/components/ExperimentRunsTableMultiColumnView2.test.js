import React from 'react';
import { shallow } from 'enzyme';
import {
  ExperimentRunsTableMultiColumnView2,
  ModelsCellRenderer,
} from './ExperimentRunsTableMultiColumnView2';
import { ColumnTypes } from '../constants';
import { RunTag } from '../sdk/MlflowMessages';
import { MemoryRouter as Router } from 'react-router-dom';

describe('ExperimentRunsTableMultiColumnView2', () => {
  let wrapper;
  let instance;
  let minimalProps;
  const exampleLoggedModelHistoryTag = {
    'mlflow.log-model.history': RunTag.fromJs({
      key: 'mlflow.log-model.history',
      value:
        '[{"run_id":"someUuid","artifact_path":"somePath",' +
        '"utc_time_created":"2020-10-22 23:18:51.726087","flavors":' +
        '{"keras":{"keras_module":"tensorflow.keras","keras_version":"2.4.0","data":"data"},' +
        '"python_function":{"loader_module":"mlflow.keras","python_version":"3.7.6",' +
        '"data":"data","env":"conda.yaml"}}}]',
    }),
  };

  beforeEach(() => {
    minimalProps = {
      runInfos: [],
      paramsList: [],
      metricsList: [],
      paramKeyList: [],
      metricKeyList: [],
      visibleTagKeyList: [],
      tagsList: [],
      onSelectionChange: jest.fn(),
      onExpand: jest.fn(),
      onSortBy: jest.fn(),
      runsSelected: {},
      runsExpanded: {},
      handleLoadMoreRuns: jest.fn(),
      loadingMore: false,
      isLoading: false,
      categorizedUncheckedKeys: {
        [ColumnTypes.ATTRIBUTES]: [],
        [ColumnTypes.PARAMS]: [],
        [ColumnTypes.METRICS]: [],
        [ColumnTypes.TAGS]: [],
      },
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ExperimentRunsTableMultiColumnView2 {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should not refit column size if there is an open column group', () => {
    wrapper = shallow(<ExperimentRunsTableMultiColumnView2 {...minimalProps} />);
    instance = wrapper.instance();
    instance.columnApi = {
      getColumnGroupState: jest.fn(() => [{ open: true }]),
    };
    instance.gridApi = {
      sizeColumnsToFit: jest.fn(),
    };
    instance.handleColumnSizeRefit();
    expect(instance.gridApi.sizeColumnsToFit).not.toBeCalled();

    instance.columnApi.getColumnGroupState = jest.fn(() => [{ open: false }]);
    instance.handleColumnSizeRefit();
    expect(instance.gridApi.sizeColumnsToFit).toBeCalled();
  });

  test('should show placeholder in logged model cells when no logged models', () => {
    const props = {
      data: {
        runInfo: {},
        tags: {},
        modelVersionsByRunUuid: {},
      },
    };
    const output = ModelsCellRenderer(props);
    expect(output).toEqual('-');
  });

  test('should show only logged model link if no registered models', () => {
    const props = {
      data: {
        runInfo: { run_uuid: 'someUuid' },
        tags: exampleLoggedModelHistoryTag,
        modelVersionsByRunUuid: {},
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('keras');
    expect(wrapper.find('img[data-test-id="logged-model-icon"]')).toHaveLength(1);
    expect(wrapper.find('img[data-test-id="registered-model-icon"]')).toHaveLength(0);
    expect(wrapper.html()).not.toContain('1 more');
  });

  test('should show both links if registered models', () => {
    const props = {
      data: {
        runInfo: { run_uuid: 'someUuid' },
        tags: exampleLoggedModelHistoryTag,
        modelVersionsByRunUuid: {
          someUuid: [
            {
              name: 'someName',
              source: 'dbfs/someUuid/artifacts/somePath',
              version: 2,
            },
          ],
        },
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('keras');
    expect(wrapper.find('img[data-test-id="logged-model-icon"]')).toHaveLength(1);
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
    const runArtifactPath = 'somePath////subdir///';
    const loggedModelHistoryTag = {
      'mlflow.log-model.history': RunTag.fromJs({
        key: 'mlflow.log-model.history',
        value:
          `[{"run_id":"someUuid","artifact_path":"${runArtifactPath}",` +
          '"utc_time_created":"2020-10-22 23:18:51.726087","flavors":' +
          '{"keras":{"keras_module":"tensorflow.keras","keras_version":"2.4.0","data":"data"},' +
          '"python_function":{"loader_module":"mlflow.keras","python_version":"3.7.6",' +
          '"data":"data","env":"conda.yaml"}}}]',
      }),
    };

    const props = {
      data: {
        runInfo: { run_uuid: 'someUuid' },
        tags: loggedModelHistoryTag,
        modelVersionsByRunUuid: {
          someUuid: [
            {
              name: 'someName',
              source: 'dbfs:///someUuid/artifacts///somePath///subdir/',
              version: 2,
            },
          ],
        },
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('keras');
    expect(wrapper.find('img[data-test-id="logged-model-icon"]')).toHaveLength(1);
    expect(wrapper.find('img[data-test-id="registered-model-icon"]')).toHaveLength(1);
  });

  test('should show 1 more if two logged models', () => {
    const tag = {
      'mlflow.log-model.history': RunTag.fromJs({
        key: 'mlflow.log-model.history',
        value:
          '[{"run_id":"someUuid","artifact_path":"somePath",' +
          '"utc_time_created":"2020-10-22 23:18:51.726087","flavors":' +
          '{"keras":{"keras_module":"tensorflow.keras","keras_version":"2.4.0","data":"data"},' +
          '"python_function":{"loader_module":"mlflow.keras","python_version":"3.7.6",' +
          '"data":"data","env":"conda.yaml"}}},' +
          '{"run_id":"someUuid","artifact_path":"someOtherPath",' +
          '"utc_time_created":"2020-10-22 23:18:51.726087","flavors":' +
          '{"keras":{"keras_module":"tensorflow.keras","keras_version":"2.4.0","data":"data"},' +
          '"python_function":{"loader_module":"mlflow.keras","python_version":"3.7.6",' +
          '"data":"data","env":"conda.yaml"}}}]',
      }),
    };
    const props = {
      data: {
        runInfo: { run_uuid: 'someUuid' },
        tags: tag,
        modelVersionsByRunUuid: {},
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('1 more');
  });
});
