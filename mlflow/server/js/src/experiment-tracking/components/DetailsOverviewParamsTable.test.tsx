import { keyBy } from 'lodash';
import { renderWithIntl, fastFillInput, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { KeyValueEntity } from '../../common/types';
import { DetailsOverviewParamsTable } from './DetailsOverviewParamsTable';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import userEvent from '@testing-library/user-event';

const testRunUuid = 'test-run-uuid';

// Larger timeout for integration testing (table rendering)
// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000);

// Generates array of param_a1, param_a2, ..., param_b2, ..., param_c3 param keys with values "value_1.0"..."value_9.0"
const sampleLatestParameters = keyBy(
  ['a', 'b', 'c'].flatMap((letter, letterIndex) =>
    [1, 2, 3].map((digit, digitIndex) => ({
      key: `param_${letter}${digit}`,
      value: 'value_' + (letterIndex * 3 + digitIndex + 1).toFixed(1),
    })),
  ),
  'key',
) as any;

describe('DetailsOverviewParamsTable', () => {
  const renderComponent = (params: Record<string, KeyValueEntity> = sampleLatestParameters) => {
    return renderWithIntl(
      <MemoryRouter>
        <DetailsOverviewParamsTable params={params} />
      </MemoryRouter>,
    );
  };

  test('Renders the table with no params recorded', () => {
    renderComponent({});
    expect(screen.getByText('No parameters recorded')).toBeInTheDocument();
  });

  test('Renders the table with values and filters values', async () => {
    renderComponent();
    expect(screen.getByRole('heading', { name: 'Parameters (9)' })).toBeInTheDocument();
    expect(screen.getByRole('row', { name: 'param_a1 value_1.0' })).toBeInTheDocument();
    expect(screen.getByRole('row', { name: 'param_c3 value_9.0' })).toBeInTheDocument();

    expect(screen.getAllByRole('row')).toHaveLength(10); // 9 rows + 1 header row

    await fastFillInput(screen.getByRole('textbox'), 'param_a');

    expect(screen.getAllByRole('row')).toHaveLength(4); // 3 rows + 1 header row

    await userEvent.clear(screen.getByRole('textbox'));
    await fastFillInput(screen.getByRole('textbox'), 'pArAM_a');

    expect(screen.getAllByRole('row')).toHaveLength(4); // 3 rows + 1 header row

    await userEvent.clear(screen.getByRole('textbox'));
    await fastFillInput(screen.getByRole('textbox'), 'param_xyz');

    expect(screen.getByText('No parameters match the search filter')).toBeInTheDocument();

    await userEvent.clear(screen.getByRole('textbox'));
    await fastFillInput(screen.getByRole('textbox'), '9.0');

    expect(screen.getAllByRole('row')).toHaveLength(2); // 1 row + 1 header row
  });
});
