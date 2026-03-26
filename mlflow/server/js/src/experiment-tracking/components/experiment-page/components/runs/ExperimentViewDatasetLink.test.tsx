import { describe, test, expect } from '@jest/globals';
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

const testRunTags = {} as Record<string, { key: string; value: string }>;

const renderComponent = (datasetWithTags: RunDatasetWithTags, experimentId?: string) => {
  return renderWithIntl(
    <MemoryRouter>
      <DesignSystemProvider>
        <ExperimentViewDatasetLink
          datasetWithTags={datasetWithTags}
          runTags={testRunTags}
          experimentId={experimentId}
        />
      </DesignSystemProvider>
    </MemoryRouter>,
  );
};

describe('ExperimentViewDatasetLink', () => {
  test('renders link to datasets tab when experimentId is provided', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.HTTP, JSON.stringify({ url: 'https://example.com/data' }));
    renderComponent(dataset, '123');

    const link = screen.getByRole('link', { name: /Open dataset/i });
    expect(link).toHaveAttribute('href', expect.stringContaining('/experiments/123/datasets'));
  });

  test('renders link for any source type when experimentId is provided', () => {
    const dataset = createDatasetWithTags(
      DatasetSourceTypes.HUGGING_FACE,
      JSON.stringify({ path: 'org/dataset-name' }),
    );
    renderComponent(dataset, '456');

    const link = screen.getByRole('link', { name: /Open dataset/i });
    expect(link).toHaveAttribute('href', expect.stringContaining('/experiments/456/datasets'));
  });

  test('renders copy button for S3 source type without experimentId', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.S3, JSON.stringify({ uri: 's3://bucket/path' }));
    renderComponent(dataset);

    expect(screen.getByText(/Copy S3 URI to clipboard/i)).toBeInTheDocument();
  });

  test('renders nothing for unknown source type without experimentId', () => {
    const dataset = createDatasetWithTags('unknown', JSON.stringify({ url: 'https://example.com' }));
    const { container } = renderComponent(dataset);
    expect(container.innerHTML).toBe('');
  });
});
