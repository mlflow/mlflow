import { IntlProvider } from 'react-intl';
import { act, render, screen } from '../../../../../common/utils/TestUtils.react18';
import type { RunDatasetWithTags } from '../../../../types';
import { DatasetSourceTypes } from '../../../../types';
import { ExperimentViewDatasetDrawer } from './ExperimentViewDatasetDrawer';
import { DesignSystemProvider } from '@databricks/design-system';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import userEvent from '@testing-library/user-event';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';

const httpDataset = {
  sourceType: DatasetSourceTypes.HTTP,
  profile: 'null',
  digest: 'abcdef',
  name: 'test_dataset_name',
  schema: '{}',
  source: '{"url":"http://test.com/something.csv"}',
};

const s3Dataset = {
  sourceType: DatasetSourceTypes.S3,
  profile: 'null',
  digest: 'abcdef',
  name: 'test_dataset_name',
  schema: '{}',
  source: '{"uri":"s3://some-bucket/hello"}',
};

const huggingFaceDataset = {
  sourceType: DatasetSourceTypes.HUGGING_FACE,
  profile: 'null',
  digest: 'abcdef',
  name: 'test_dataset_name',
  schema: '{}',
  source: '{"path":"databricks/databricks-dolly-15k"}',
};

describe('ExperimentViewDatasetDrawer', () => {
  let navigatorClipboard: Clipboard;

  // Prepare fake clipboard
  beforeAll(() => {
    navigatorClipboard = navigator.clipboard;
    (navigator.clipboard as any) = { writeText: jest.fn() };
  });

  // Cleanup and restore clipboard
  afterAll(() => {
    (navigator.clipboard as any) = navigatorClipboard;
  });

  const renderTestComponent = ({ dataset }: { dataset: RunDatasetWithTags['dataset'] }) => {
    return render(
      <ExperimentViewDatasetDrawer
        isOpen
        setIsOpen={() => {}}
        selectedDatasetWithRun={{
          datasetWithTags: {
            dataset,
            tags: [],
          },
          runData: {
            runUuid: 'runUuid',
            datasets: [],
          },
        }}
        setSelectedDatasetWithRun={() => {}}
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <MemoryRouter>
              <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
                <DesignSystemProvider>{children}</DesignSystemProvider>
              </MockedReduxStoreProvider>
            </MemoryRouter>
          </IntlProvider>
        ),
      },
    );
  };
  test('it renders HTTP dataset source type', () => {
    renderTestComponent({ dataset: httpDataset });
    expect(screen.getByText('Source type: HTTP')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'http://test.com/something.csv' })).toBeInTheDocument();
  });

  test('it renders S3 dataset source type', async () => {
    renderTestComponent({ dataset: s3Dataset });
    expect(screen.getByText('Source type: S3')).toBeInTheDocument();
    const copyButton = screen.getByRole('button', { name: /Copy S3 URI/ });
    expect(copyButton).toBeInTheDocument();
    await userEvent.click(copyButton);
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('s3://some-bucket/hello');
  });

  test('it renders Hugging Face dataset source type', () => {
    renderTestComponent({ dataset: huggingFaceDataset });
    expect(screen.getByText('Source type: Hugging Face')).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'https://huggingface.co/datasets/databricks/databricks-dolly-15k' }),
    ).toBeInTheDocument();
  });
});
