import { jest, describe, beforeEach, it, expect, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { getTableRowByCellText, getTableRows } from '@databricks/design-system/test-utils/rtl';
import { renderWithIntl, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { ModelListTableProps } from './ModelListTable';
import { ModelListTable } from './ModelListTable';
import { DesignSystemProvider } from '@databricks/design-system';
import { Stages } from '../../constants';
import Utils from '../../../common/utils/Utils';
import { withNextModelsUIContext } from '../../hooks/useNextModelsUI';
import { ModelsNextUIToggleSwitch } from '../ModelsNextUIToggleSwitch';
import { shouldShowModelsNextUI } from '../../../common/utils/FeatureUtils';
import { I18nUtils } from '../../../i18n/I18nUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldShowModelsNextUI: jest.fn(),
}));

const MODELS = [
  {
    name: 'test_model_1',
    creation_timestamp: 1234,
    email_subscription_status: 'active',
    last_updated_timestamp: 1234,
    permission_level: '',
    source: 'notebook',
    status: 'active',
    user_id: '123',
    latest_versions: [{ version: 1 } as any],
    tags: [],
  },
  {
    name: 'test_model_2',
    creation_timestamp: 100000,
    email_subscription_status: 'active',
    last_updated_timestamp: 100000,
    permission_level: '',
    source: 'notebook',
    status: 'active',
    user_id: '123',
    latest_versions: [
      { version: 2, current_stage: Stages.PRODUCTION },
      { version: 3, current_stage: Stages.STAGING },
      { version: 1 },
    ] as any,
    tags: [],
  },
];

describe('ModelListTable', () => {
  const minimalProps: ModelListTableProps = {
    isLoading: false,
    error: undefined,
    modelsData: MODELS as any,
    onSortChange: jest.fn(),
    orderByAsc: false,
    orderByKey: 'name',
    pagination: <div data-testid="pagination" />,
    isFiltered: false,
  };

  const renderTable = (moreProps: Partial<ModelListTableProps> = {}) =>
    renderWithIntl(
      <MemoryRouter>
        <DesignSystemProvider>
          <ModelListTable {...minimalProps} {...moreProps} />
        </DesignSystemProvider>
      </MemoryRouter>,
    );

  it('renders a header row plus one row per model, and the pagination slot', () => {
    renderTable();
    // One header row and two data rows.
    expect(screen.getAllByRole('row')).toHaveLength(3);
    expect(screen.getByTestId('pagination')).toBeInTheDocument();
  });

  it('renders the modification date column', () => {
    renderTable();
    const intl = I18nUtils.createIntlWithLocale();
    const [firstRow] = getTableRows(screen.getByRole('table')).bodyRows;
    expect(firstRow).toHaveTextContent(Utils.formatTimestamp(MODELS[0].last_updated_timestamp, intl));
  });

  it('renders the model links', () => {
    renderTable();
    expect(
      within(getTableRowByCellText(screen.getByRole('table'), 'test_model_1')).getByRole('link', {
        name: 'test_model_1',
      }),
    ).toHaveAttribute('href', expect.stringMatching(/\/models\/test_model_1$/));
    expect(
      within(getTableRowByCellText(screen.getByRole('table'), 'test_model_2')).getByRole('link', {
        name: 'test_model_2',
      }),
    ).toHaveAttribute('href', expect.stringMatching(/\/models\/test_model_2$/));
  });

  it('renders the simple model version link', () => {
    renderTable();
    const row = getTableRowByCellText(screen.getByRole('table'), 'test_model_1');
    expect(within(row).getByRole('link', { name: 'Version 1' })).toHaveAttribute(
      'href',
      expect.stringMatching(/\/models\/test_model_1\/versions\/1$/),
    );
  });

  it('renders staged model version links but not the unstaged version', () => {
    renderTable();
    // Model #2 shows versions 2 (Production) and 3 (Staging), but not version 1. Match by href, since
    // the same "Version N" link text can appear in more than one column (e.g. latest + staged).
    const row = getTableRowByCellText(screen.getByRole('table'), 'test_model_2');
    const hrefs = within(row)
      .getAllByRole('link')
      .map((link) => link.getAttribute('href'));
    expect(hrefs).not.toContainEqual(expect.stringMatching(/\/models\/test_model_2\/versions\/1$/));
    expect(hrefs).toContainEqual(expect.stringMatching(/\/models\/test_model_2\/versions\/2$/));
    expect(hrefs).toContainEqual(expect.stringMatching(/\/models\/test_model_2\/versions\/3$/));
  });

  it('renders tags and expands/collapses the overflow', async () => {
    const modelWithManyTags = {
      ...MODELS[0],
      tags: [
        ...new Array(4).fill(0).map((_, index) => ({ key: `Tag ${index + 1}`, value: `Value ${index + 1}` })),
        { key: 'Empty tag', value: undefined },
      ],
    };
    renderTable({ modelsData: [modelWithManyTags as any] });
    const row = getTableRowByCellText(screen.getByRole('table'), MODELS[0].name);

    expect(row).toHaveTextContent('Tag 1: Value 1');
    expect(row).toHaveTextContent('Tag 2: Value 2');
    expect(row).toHaveTextContent('Tag 3: Value 3');
    expect(row).not.toHaveTextContent('Tag 4: Value 4');

    await userEvent.click(within(row).getByRole('button', { name: '2 more' }));
    expect(row).toHaveTextContent('Tag 4: Value 4');
    expect(row).toHaveTextContent('Empty tag: (empty)');

    await userEvent.click(within(row).getByRole('button', { name: 'Show less' }));
    expect(row).not.toHaveTextContent('Tag 4: Value 4');
  });

  test('shows the empty-results message when filtered results are empty', () => {
    renderTable({ modelsData: [], isFiltered: true });
    expect(
      screen.getByText('No results. Try using a different keyword or adjusting your filters.'),
    ).toBeInTheDocument();
  });

  test('shows an error message on API errors', () => {
    const errMsg = 'TEMPORARILY_UNAVAILABLE: Backend unavailable';
    renderTable({ error: new Error(errMsg) });
    expect(screen.getByText(/Error fetching models/)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(errMsg))).toBeInTheDocument();
  });
});
