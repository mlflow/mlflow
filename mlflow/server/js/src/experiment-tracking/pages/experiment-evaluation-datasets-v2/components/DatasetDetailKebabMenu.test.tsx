// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { workspaceFetch } from '@databricks/web-shared/spog/workspace-console';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { DatasetDetailKebabMenu } from './DatasetDetailKebabMenu';
import type { DatasetNotifyApi } from '../hooks/useDatasetNotifications';
import { renderDatasetsPage } from '../test-utils/renderDatasetsPage';
import { mockEmptyResponse } from '../test-utils/mockResponses';

jest.mock('@databricks/web-shared/spog/workspace-console', () => ({
  workspaceFetch: jest.fn(),
}));

const notifyStub: DatasetNotifyApi = {
  success: jest.fn(),
  error: jest.fn(),
};

const baseDataset: Dataset = {
  dataset_id: 'ds-1',
  create_time: '2026-01-01T00:00:00Z',
  name: 'main.evals.support_qa',
};

describe('DatasetDetailKebabMenu', () => {
  // setupTestRouter registers beforeAll/afterAll/beforeEach hooks — must live at describe scope.
  const { history } = setupTestRouter();

  beforeEach(() => {
    jest.mocked(workspaceFetch).mockResolvedValue(mockEmptyResponse());
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  const mountKebab = (dataset: Dataset) =>
    renderDatasetsPage({
      initialUrl: '/test',
      routes: [
        {
          path: '/test',
          element: <DatasetDetailKebabMenu experimentId="exp-1" dataset={dataset} notify={notifyStub} />,
        },
      ],
      history,
    });

  test('renders "View in Unity Catalog" as an external link when dataset.name is a 3-part UC path', async () => {
    const user = userEvent.setup();
    mountKebab(baseDataset);

    await user.click(await screen.findByRole('button', { name: /Dataset actions/i }));

    const link = await screen.findByRole('menuitem', { name: /View in Unity Catalog/i });
    expect(link).toHaveAttribute('href', '/explore/data/main/evals/support_qa');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });

  test('omits "View in Unity Catalog" when dataset.name is not a 3-part UC path', async () => {
    const user = userEvent.setup();
    mountKebab({ ...baseDataset, name: 'not-a-uc-path' });

    await user.click(await screen.findByRole('button', { name: /Dataset actions/i }));
    expect(screen.queryByRole('menuitem', { name: /View in Unity Catalog/i })).not.toBeInTheDocument();
  });

  test('"View dataset metadata" is disabled when digest, schema, and profile are all empty', async () => {
    const user = userEvent.setup();
    mountKebab(baseDataset);

    await user.click(await screen.findByRole('button', { name: /Dataset actions/i }));
    const item = await screen.findByRole('menuitem', { name: /View dataset metadata/i });
    expect(item).toHaveAttribute('aria-disabled', 'true');
  });

  test('"View dataset metadata" opens a modal showing digest, schema, and profile sections', async () => {
    const user = userEvent.setup();
    mountKebab({
      ...baseDataset,
      digest: 'abc123def456',
      schema: '{"columns":[{"name":"q","type":"string"}]}',
      profile: '{"row_count":42}',
    });

    await user.click(await screen.findByRole('button', { name: /Dataset actions/i }));
    await user.click(await screen.findByRole('menuitem', { name: /View dataset metadata/i }));

    const dialog = await screen.findByRole('dialog', { name: /Dataset metadata/i });
    expect(within(dialog).getByText('abc123def456')).toBeInTheDocument();
    // Schema and profile JSON blobs are rendered into pretty-printed <pre> blocks.
    expect(within(dialog).getByText(/"columns"/)).toBeInTheDocument();
    expect(within(dialog).getByText(/"row_count"/)).toBeInTheDocument();
  });

  test('metadata modal hides empty sections', async () => {
    const user = userEvent.setup();
    mountKebab({ ...baseDataset, digest: 'only-a-digest' });

    await user.click(await screen.findByRole('button', { name: /Dataset actions/i }));
    await user.click(await screen.findByRole('menuitem', { name: /View dataset metadata/i }));

    const dialog = await screen.findByRole('dialog', { name: /Dataset metadata/i });
    expect(within(dialog).getByText('only-a-digest')).toBeInTheDocument();
    // No Schema or Profile section labels when those fields are missing.
    expect(within(dialog).queryByText('Schema')).not.toBeInTheDocument();
    expect(within(dialog).queryByText('Profile')).not.toBeInTheDocument();
  });

  test('metadata modal renders raw text when a field is not valid JSON', async () => {
    const user = userEvent.setup();
    mountKebab({ ...baseDataset, schema: 'this-is-not-json' });

    await user.click(await screen.findByRole('button', { name: /Dataset actions/i }));
    await user.click(await screen.findByRole('menuitem', { name: /View dataset metadata/i }));

    const dialog = await screen.findByRole('dialog', { name: /Dataset metadata/i });
    expect(within(dialog).getByText('this-is-not-json')).toBeInTheDocument();
  });

  test('clicking Close dismisses the metadata modal', async () => {
    const user = userEvent.setup();
    mountKebab({ ...baseDataset, digest: 'd' });

    await user.click(await screen.findByRole('button', { name: /Dataset actions/i }));
    await user.click(await screen.findByRole('menuitem', { name: /View dataset metadata/i }));
    const dialog = await screen.findByRole('dialog', { name: /Dataset metadata/i });

    await user.click(within(dialog).getByRole('button', { name: /Done/i }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /Dataset metadata/i })).not.toBeInTheDocument());
  });
});