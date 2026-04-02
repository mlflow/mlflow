import { describe, test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ExperimentViewDatasetLink } from './ExperimentViewDatasetLink';
import { DatasetSourceTypes } from '../../../../types';
import type { RunDatasetWithTags } from '../../../../types';
import { DesignSystemProvider } from '@databricks/design-system';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';

const createDatasetWithTags = (sourceType: string, source: string): RunDatasetWithTags =>
  ({
    dataset: {
      name: 'test-dataset',
      digest: 'abc123',
      sourceType,
      source,
    },
    tags: [],
  }) as any;

const renderComponent = (datasetWithTags: RunDatasetWithTags) => {
  return renderWithIntl(
    <MemoryRouter>
      <DesignSystemProvider>
        <ExperimentViewDatasetLink datasetWithTags={datasetWithTags} />
      </DesignSystemProvider>
    </MemoryRouter>,
  );
};

describe('ExperimentViewDatasetLink', () => {
  let windowOpenSpy: jest.SpiedFunction<typeof window.open>;

  beforeEach(() => {
    windowOpenSpy = jest.spyOn(window, 'open').mockReturnValue(null);
  });

  afterEach(() => {
    windowOpenSpy.mockRestore();
  });

  test('renders copy button for S3 source type', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.S3, JSON.stringify({ uri: 's3://bucket/path' }));
    renderComponent(dataset);

    expect(screen.getByText(/Copy S3 URI to clipboard/i)).toBeInTheDocument();
  });

  test('renders open dataset button for HTTP source type and opens URL on click', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.HTTP, JSON.stringify({ url: 'https://example.com/data' }));
    renderComponent(dataset);

    const button = screen.getByText(/Open dataset/i);
    expect(button).toBeInTheDocument();
    button.click();
    expect(windowOpenSpy).toHaveBeenCalledWith('https://example.com/data', '_blank', 'noopener,noreferrer');
  });

  test('renders open dataset button for Hugging Face source type and opens URL on click', () => {
    const dataset = createDatasetWithTags(
      DatasetSourceTypes.HUGGING_FACE,
      JSON.stringify({ path: 'org/dataset-name' }),
    );
    renderComponent(dataset);

    const button = screen.getByText(/Open dataset/i);
    expect(button).toBeInTheDocument();
    button.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      'https://huggingface.co/datasets/org/dataset-name',
      '_blank',
      'noopener,noreferrer',
    );
  });

  test('renders nothing for local source type', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.LOCAL, JSON.stringify({ uri: '/data/photos/my-concept' }));
    const { container } = renderComponent(dataset);
    expect(container.innerHTML).toBe('');
    expect(windowOpenSpy).not.toHaveBeenCalled();
  });

  test('renders nothing for unknown source type', () => {
    const dataset = createDatasetWithTags('unknown', JSON.stringify({ url: 'https://example.com' }));
    const { container } = renderComponent(dataset);
    expect(container.innerHTML).toBe('');
    expect(windowOpenSpy).not.toHaveBeenCalled();
  });
});
