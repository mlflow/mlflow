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
  let minimalProps;
  const runTags = {
    'mlflow.log-model.history': RunTag.fromJs({
      key: 'mlflow.log-model.history',
      value: JSON.stringify([
        {
          run_id: 'run-uuid',
          artifact_path: 'somePath',
          utc_time_created: '2020-10-22',
          flavors: { keras: {}, python_function: {} },
        },
      ]),
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
        runInfo: { run_uuid: 'run-uuid' },
        tags: runTags,
        modelVersionsByRunUuid: {},
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
        data: {
          runInfo: { run_uuid: 'runUuid', experiment_id: 'experimentId' },
          tags: runTags,
          modelVersionsByRunUuid: modelVersions,
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
    tester({}, 'experiments/experimentId/runs/runUuid/artifactPath/somePath', '.logged-model-link');

    tester(
      {
        runUuid: [
          {
            name: 'someModel',
            source: 'dbfs/runUuid/artifacts/somePath',
            version: 2,
          },
        ],
      },
      'models/someModel/versions/2',
      '.registered-model-link',
    );
  });

  test('should only show registered model link if registered model', () => {
    const props = {
      data: {
        runInfo: { run_uuid: 'someUuid' },
        tags: runTags,
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
    const runArtifactPath = 'somePath////subdir///';
    const tags = {
      'mlflow.log-model.history': RunTag.fromJs({
        key: 'mlflow.log-model.history',
        value: JSON.stringify([
          {
            run_id: 'run-uuid',
            artifact_path: `${runArtifactPath}`,
            utc_time_created: '2020-10-22',
            flavors: { keras: {}, python_function: {} },
          },
        ]),
      }),
    };

    const props = {
      data: {
        runInfo: { run_uuid: 'someUuid' },
        tags: tags,
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
    expect(wrapper.html()).toContain('someName');
    expect(wrapper.find('img[data-test-id="logged-model-icon"]')).toHaveLength(0);
    expect(wrapper.find('img[data-test-id="registered-model-icon"]')).toHaveLength(1);
  });

  test('should show 1 more if two logged models', () => {
    const tags = {
      'mlflow.log-model.history': RunTag.fromJs({
        key: 'mlflow.log-model.history',
        value: JSON.stringify([
          {
            run_id: 'run-uuid',
            artifact_path: 'somePath',
            utc_time_created: '2020-10-22',
            flavors: { keras: {}, python_function: {} },
          },
          {
            run_id: 'run-uuid',
            artifact_path: 'someOtherPath',
            utc_time_created: '2020-10-22',
            flavors: { keras: {}, python_function: {} },
          },
        ]),
      }),
    };
    const props = {
      data: {
        runInfo: { run_uuid: 'run-uuid' },
        tags: tags,
        modelVersionsByRunUuid: {},
      },
    };
    const output = ModelsCellRenderer(props);
    wrapper = shallow(<Router>{output}</Router>);
    expect(wrapper.html()).toContain('1 more');
  });
});
