import { keyBy } from 'lodash';
import { renderWithIntl, fastFillInput, act, screen } from 'common/utils/TestUtils.react17';
import { KeyValueEntity } from '../../../types';
import { RunViewParamsTable } from './RunViewParamsTable';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';

const testRunUuid = 'test-run-uuid';

// Larger timeout for integration testing (table rendering)
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

describe('RunViewParamsTable', () => {
  const renderComponent = (params: Record<string, KeyValueEntity> = sampleLatestParameters) => {
    return renderWithIntl(
      <MemoryRouter>
        <RunViewParamsTable runUuid={testRunUuid} params={params} />
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

    await act(async () => {
      fastFillInput(screen.getByRole('textbox'), 'param_a');
    });

    expect(screen.getAllByRole('row')).toHaveLength(4); // 3 rows + 1 header row

    await act(async () => {
      fastFillInput(screen.getByRole('textbox'), 'param_xyz');
    });

    expect(screen.getByText('No parameters match the search filter')).toBeInTheDocument();
  });
});
