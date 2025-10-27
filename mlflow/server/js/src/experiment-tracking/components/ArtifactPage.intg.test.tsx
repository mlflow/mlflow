import { last } from 'lodash';
import { render, screen, waitFor } from '@testing-library/react';
import ArtifactPage from './ArtifactPage';
import { MockedReduxStoreProvider } from '../../common/utils/TestUtils';
import { MlflowService } from '../sdk/MlflowService';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { IntlProvider } from 'react-intl';
import userEvent from '@testing-library/user-event';
import { getArtifactContent, getArtifactBytesContent } from '../../common/utils/ArtifactUtils';
import { TestRouter, testRoute } from '../../common/utils/RoutingTestUtils';
import { MLFLOW_LOGGED_ARTIFACTS_TAG } from '../constants';
import Utils from '../../common/utils/Utils';
import { Services } from '../../model-registry/services';
import type { ReduxState } from '../../redux-types';
import { applyMiddleware, combineReducers, createStore, type DeepPartial } from 'redux';
import type { KeyValueEntity } from '../../common/types';
// eslint-disable-next-line import/no-nodejs-modules
import { readFileSync } from 'fs';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larger timeout for integration testing

jest.mock('../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/ArtifactUtils')>('../../common/utils/ArtifactUtils'),
  getArtifactContent: jest.fn(),
  getArtifactBytesContent: jest.fn(),
}));

jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/FeatureUtils')>('../../common/utils/FeatureUtils'),
}));

// List of various artifacts to be downloaded and rendered
const artifactTestCases: string[] = [
  // JSON files
  'json/table_eval_data.json',
  'json/table_unnamed_columns.json',
  'json/table_eval_broken.json',

  // GeoJSON files
  'geojson/sample.geojson',

  // CSV files
  'csv/csv_colons.csv',
  'csv/csv_commas.csv',
  'csv/csv_broken.csv',

  // Image files
  'png/sample_small.png',

  // HTML files
  'html/test_html.html',

  // PDF files
  'pdf/pdf_sample.pdf',
];

// A small stub that satisfies <RequestStateWrapper> by
// providing fake API responses that are always fulfilled
const alwaysFulfilledResponseApiStub = new Proxy(
  {},
  {
    get() {
      return { active: false };
    },
  },
);

/**
 * Local util that loads a fixture file from the artifact-fixtures directory
 */
const loadLocalArtifactFixtureFile = (fileName: string) =>
  readFileSync([__dirname, 'run-page/artifact-fixtures', fileName].join('/'));

/**
 * Mocks the artifact retrieval utils to return the provided artifact data
 */
const mockArtifactRetrieval = <T extends BlobPart>(artifactData: T) => {
  const getArtifactContentMocked = (_: string, isBinary = false) => {
    return new Promise((fetchArtifactResolve, reject) => {
      const blob = new Blob([artifactData]);
      const fileReader = new FileReader();

      fileReader.onload = (event) => {
        if (!event?.target?.result) {
          return reject();
        }
        fetchArtifactResolve(event.target.result);
      };

      if (isBinary) {
        fileReader.readAsArrayBuffer(blob);
      } else {
        fileReader.readAsText(blob);
      }
    });
  };
  jest.mocked(getArtifactContent).mockImplementation(getArtifactContentMocked);
  jest.mocked(getArtifactBytesContent).mockImplementation((...props) => getArtifactContentMocked(...props, true));
};

/**
 * Creates fake run tags. If a file is JSON, it will be tagged as a logged artifact.
 */
const createRunTagsForFile = (baseFileName: string): Record<string, KeyValueEntity> => {
  if (baseFileName.endsWith('.json')) {
    return {
      [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
        key: MLFLOW_LOGGED_ARTIFACTS_TAG,
        value: `[{"path": "${baseFileName}", "type": "table"}]`,
      },
    };
  }

  return {};
};

describe('Artifact page, artifact files rendering integration test', () => {
  beforeEach(() => {
    jest.spyOn(MlflowService, 'listArtifacts').mockResolvedValue({});
    jest.spyOn(Services, 'searchRegisteredModels').mockResolvedValue({ registered_models: [] });
  });
  it.each(artifactTestCases)('renders artifact file: %s', async (fileName) => {
    const fileContents = loadLocalArtifactFixtureFile(fileName);
    const baseFilename = last(fileName.split('/')) ?? '';

    const runTags = createRunTagsForFile(baseFilename);

    mockArtifactRetrieval(fileContents);

    const testReduxStoreState: DeepPartial<ReduxState> = {
      apis: alwaysFulfilledResponseApiStub,
      entities: {
        modelVersionsByModel: {},
        artifactRootUriByRunUuid: {
          'test-run-uuid': 'dbfs:/some-path/',
        },
        artifactsByRunUuid: {
          'test-run-uuid': new ArtifactNode(true, undefined, {
            [baseFilename]: new ArtifactNode(
              false,
              {
                path: baseFilename,
                is_dir: false,
                file_size: 1000,
              },
              undefined,
            ),
          }),
        },
      },
    };

    render(<ArtifactPage runUuid="test-run-uuid" runTags={runTags} experimentId="test-experiment-id" />, {
      wrapper: ({ children }) => (
        <TestRouter
          routes={[
            testRoute(
              <IntlProvider locale="en">
                <MockedReduxStoreProvider state={testReduxStoreState}>{children}</MockedReduxStoreProvider>
              </IntlProvider>,
            ),
          ]}
        />
      ),
    });

    // Wait for the artifact tree to be rendered
    await waitFor(() => {
      expect(screen.getByText(baseFilename)).toBeInTheDocument();
    });

    // Click on the artifact to open it
    await userEvent.click(screen.getByText(baseFilename));

    // Wait for the artifact to be loaded and rendered
    await waitFor(() => {
      expect(screen.getByTitle(baseFilename)).toBeInTheDocument();
      expect(screen.queryByText('Artifact loading')).not.toBeInTheDocument();
    });
  });

  it('renders model artifact', async () => {
    const fileContents = loadLocalArtifactFixtureFile('models/MLmodel');

    const runTags = {
      [Utils.loggedModelsTag]: {
        key: Utils.loggedModelsTag,
        value: `[{"artifact_path":"logged_model","signature":{"inputs":"","outputs":"","params":null},"flavors":{"python_function":{"cloudpickle_version":"2.2.1","loader_module":"mlflow.pyfunc.model","python_model":"python_model.pkl","env":{"conda":"conda.yaml","virtualenv":"python_env.yaml"},"python_version":"3.10.12"}},"run_id":"test-run-uuid","model_uuid":"test-model-uuid","utc_time_created":"2023-01-01 10:57:14.780880"}]`,
      },
    };

    mockArtifactRetrieval(fileContents);

    const testReduxStoreState: DeepPartial<ReduxState> = {
      apis: alwaysFulfilledResponseApiStub,
      entities: {
        modelVersionsByModel: {},
        artifactRootUriByRunUuid: {
          'test-run-uuid': 'dbfs:/some-path/',
        },
        artifactsByRunUuid: {
          'test-run-uuid': new ArtifactNode(true, undefined, {
            logged_model: new ArtifactNode(
              false,
              {
                path: 'logged_model',
                is_dir: true,
              },
              {
                MLmodel: new ArtifactNode(
                  false,
                  {
                    is_dir: false,
                    file_size: 1000,
                    path: 'logged_model/MLmodel',
                  },
                  undefined,
                ),
              },
            ),
          }),
        },
      },
    };

    render(<ArtifactPage runUuid="test-run-uuid" runTags={runTags} experimentId="test-experiment-id" />, {
      wrapper: ({ children }) => (
        <TestRouter
          routes={[
            testRoute(
              <IntlProvider locale="en">
                <MockedReduxStoreProvider state={testReduxStoreState}>{children}</MockedReduxStoreProvider>
              </IntlProvider>,
            ),
          ]}
        />
      ),
    });

    // Wait for the artifact tree to be rendered
    await waitFor(() => {
      expect(screen.getByLabelText('logged_model')).toBeInTheDocument();
    });

    // Click on the artifact to open it
    await userEvent.click(screen.getByLabelText('logged_model'));

    // Wait for the artifact to be loaded and rendered
    await waitFor(() => {
      expect(screen.getByText('test_model_input')).toBeInTheDocument();
      expect(screen.getByText('test_model_output')).toBeInTheDocument();
    });
  });
});

describe('Artifact page, artifact list request error handling', () => {
  beforeEach(() => {
    jest.spyOn(MlflowService, 'listArtifacts').mockResolvedValue({});
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  // Mock failed API response as a slice of redux store which ArtifactPage uses for getting error state
  const alwaysFailingResponseApiStub = new Proxy(
    {},
    {
      get() {
        return { active: false, error: new ErrorWrapper({ message: 'User does not have permissions' }, 403) };
      },
    },
  );

  const testReduxStoreState: DeepPartial<ReduxState> = {
    apis: alwaysFailingResponseApiStub,
    entities: {
      modelVersionsByModel: {},
      artifactRootUriByRunUuid: {},
      artifactsByRunUuid: {},
    },
  };

  test('renders error message when artifact list request fails', async () => {
    render(<ArtifactPage runUuid="test-run-uuid" runTags={{}} experimentId="test-experiment-id" />, {
      wrapper: ({ children }) => (
        <TestRouter
          routes={[
            testRoute(
              <IntlProvider locale="en">
                <MockedReduxStoreProvider state={testReduxStoreState}>{children}</MockedReduxStoreProvider>
              </IntlProvider>,
            ),
          ]}
        />
      ),
    });

    await waitFor(() => {
      expect(screen.getByText('Loading artifact failed')).toBeInTheDocument();
      expect(screen.getByText('User does not have permissions')).toBeInTheDocument();
    });
  });
});
