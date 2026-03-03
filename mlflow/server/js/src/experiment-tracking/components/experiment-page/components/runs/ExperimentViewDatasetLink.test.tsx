import { describe, test, expect } from '@jest/globals';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ExperimentViewDatasetLink } from './ExperimentViewDatasetLink';
import { DatasetSourceTypes } from '../../../../types';
import type { RunDatasetWithTags } from '../../../../types';
import { DesignSystemProvider } from '@databricks/design-system';

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

const testRunTags = {} as Record<string, { key: string; value: string }>;

const renderComponent = (datasetWithTags: RunDatasetWithTags) => {
  return renderWithIntl(
    <DesignSystemProvider>
      <ExperimentViewDatasetLink datasetWithTags={datasetWithTags} runTags={testRunTags} />
    </DesignSystemProvider>,
  );
};

describe('ExperimentViewDatasetLink', () => {
  test('renders clickable link for HTTP source type', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.HTTP, JSON.stringify({ url: 'https://example.com/data' }));
    renderComponent(dataset);

    const link = screen.getByRole('link', { name: /Open dataset/i });
    expect(link).toHaveAttribute('href', 'https://example.com/data');
    expect(link).toHaveAttribute('target', '_blank');
  });

  test('renders clickable link for EXTERNAL source type with url', () => {
    const dataset = createDatasetWithTags(
      DatasetSourceTypes.EXTERNAL,
      JSON.stringify({ url: 'https://external.example.com/dataset' }),
    );
    renderComponent(dataset);

    const link = screen.getByRole('link', { name: /Open dataset/i });
    expect(link).toHaveAttribute('href', 'https://external.example.com/dataset');
    expect(link).toHaveAttribute('target', '_blank');
  });

  test('renders nothing for EXTERNAL source type without url', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.EXTERNAL, JSON.stringify({ other: 'value' }));
    const { container } = renderComponent(dataset);
    expect(container.innerHTML).toBe('');
  });

  test('renders clickable link for HUGGING_FACE source type', () => {
    const dataset = createDatasetWithTags(
      DatasetSourceTypes.HUGGING_FACE,
      JSON.stringify({ path: 'org/dataset-name' }),
    );
    renderComponent(dataset);

    const link = screen.getByRole('link', { name: /Open dataset/i });
    expect(link).toHaveAttribute('href', 'https://huggingface.co/datasets/org/dataset-name');
  });

  test('renders copy button for S3 source type', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.S3, JSON.stringify({ uri: 's3://bucket/path' }));
    renderComponent(dataset);

    expect(screen.getByText(/Copy S3 URI to clipboard/i)).toBeInTheDocument();
  });

  test('renders nothing for unknown source type', () => {
    const dataset = createDatasetWithTags('unknown', JSON.stringify({ url: 'https://example.com' }));
    const { container } = renderComponent(dataset);
    expect(container.innerHTML).toBe('');
  });
});
