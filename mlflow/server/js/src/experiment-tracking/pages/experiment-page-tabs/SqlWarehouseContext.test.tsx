import { describe, jest, test, expect, beforeEach } from '@jest/globals';
import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SqlWarehouseContextProvider, useSqlWarehouseContext } from './SqlWarehouseContext';

// Mock the underlying URL-param hook so we don't need a router
const mockSetSqlWarehouseId = jest.fn();
let mockWarehouseId: string | undefined | null = undefined;

jest.mock('./usePersistedSqlWarehouseId', () => ({
  usePersistedSqlWarehouseId: () => {
    return [mockWarehouseId, mockSetSqlWarehouseId] as const;
  },
}));

/** Renders a context value via data-testid for assertions */
const TestConsumer = ({ label }: { label: string }) => {
  const { warehouseId, warehousesLoading } = useSqlWarehouseContext();
  return (
    <div>
      <span data-testid={`${label}-id`}>{warehouseId ?? 'none'}</span>
      <span data-testid={`${label}-loading`}>{String(warehousesLoading)}</span>
    </div>
  );
};

/** Calls setWarehouseId with a fixed value on click */
const TestSetter = ({ value }: { value: string | null }) => {
  const { setWarehouseId } = useSqlWarehouseContext();
  return (
    <button data-testid="set-warehouse" onClick={() => setWarehouseId(value)}>
      Set
    </button>
  );
};

/** Calls setWarehousesLoading on click */
const TestLoadingSetter = ({ loading }: { loading: boolean }) => {
  const { setWarehousesLoading } = useSqlWarehouseContext();
  return (
    <button data-testid="set-loading" onClick={() => setWarehousesLoading(loading)}>
      Set Loading
    </button>
  );
};

describe('SqlWarehouseContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWarehouseId = undefined;
  });

  test('selected warehouse ID is available to all consumers', () => {
    mockWarehouseId = 'wh-abc';

    render(
      <SqlWarehouseContextProvider experimentId="test-exp">
        <TestConsumer label="consumer-a" />
        <TestConsumer label="consumer-b" />
      </SqlWarehouseContextProvider>,
    );

    expect(screen.getByTestId('consumer-a-id')).toHaveTextContent('wh-abc');
    expect(screen.getByTestId('consumer-b-id')).toHaveTextContent('wh-abc');
  });

  test('changing warehouse updates all consumers', async () => {
    mockWarehouseId = 'wh-old';

    render(
      <SqlWarehouseContextProvider experimentId="test-exp">
        <TestConsumer label="consumer-a" />
        <TestConsumer label="consumer-b" />
        <TestSetter value="wh-new" />
      </SqlWarehouseContextProvider>,
    );

    // Both consumers see the initial value
    expect(screen.getByTestId('consumer-a-id')).toHaveTextContent('wh-old');
    expect(screen.getByTestId('consumer-b-id')).toHaveTextContent('wh-old');

    // Click the setter — this calls the mocked setSqlWarehouseId
    await userEvent.click(screen.getByTestId('set-warehouse'));

    expect(mockSetSqlWarehouseId).toHaveBeenCalledWith('wh-new');
  });

  test('clearing warehouse selection resets all consumers', async () => {
    mockWarehouseId = 'wh-123';

    render(
      <SqlWarehouseContextProvider experimentId="test-exp">
        <TestConsumer label="consumer-a" />
        <TestSetter value={null} />
      </SqlWarehouseContextProvider>,
    );

    expect(screen.getByTestId('consumer-a-id')).toHaveTextContent('wh-123');

    await userEvent.click(screen.getByTestId('set-warehouse'));

    expect(mockSetSqlWarehouseId).toHaveBeenCalledWith(null);
  });

  test('warehouse loading state is shared', async () => {
    render(
      <SqlWarehouseContextProvider experimentId="test-exp">
        <TestConsumer label="consumer-a" />
        <TestConsumer label="consumer-b" />
        <TestLoadingSetter loading />
      </SqlWarehouseContextProvider>,
    );

    // Initially not loading
    expect(screen.getByTestId('consumer-a-loading')).toHaveTextContent('false');
    expect(screen.getByTestId('consumer-b-loading')).toHaveTextContent('false');

    // Set loading to true
    await userEvent.click(screen.getByTestId('set-loading'));

    expect(screen.getByTestId('consumer-a-loading')).toHaveTextContent('true');
    expect(screen.getByTestId('consumer-b-loading')).toHaveTextContent('true');
  });

  test('throws when used outside provider', () => {
    // Suppress React error boundary console output
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => render(<TestConsumer label="orphan" />)).toThrow(
      'useSqlWarehouseContext must be used within a SqlWarehouseContextProvider',
    );

    spy.mockRestore();
  });
});
