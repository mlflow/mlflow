import userEvent from '@testing-library/user-event-14';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import { render, screen } from '../../../../common/utils/TestUtils.react18';
import { useChartMoveUpDownFunctions, useOrderedCharts } from './useOrderedCharts';

describe('useChartMoveUpDownFunctions', () => {
  test('it should properly allow moving up/down', () => {
    const onReorder = jest.fn();
    const { canMoveDown, canMoveUp, moveChartDown, moveChartUp } = useChartMoveUpDownFunctions(
      ['met1', 'met2', 'met3'],
      onReorder,
    );

    expect(canMoveUp('met3')).toEqual(true);
    expect(canMoveDown('met3')).toEqual(false);

    expect(canMoveUp('met1')).toEqual(false);
    expect(canMoveDown('met1')).toEqual(true);

    expect(canMoveUp('met2')).toEqual(true);
    expect(canMoveDown('met2')).toEqual(true);

    moveChartUp('met3');
    expect(onReorder).toHaveBeenLastCalledWith('met3', 'met2');

    moveChartUp('met2');
    expect(onReorder).toHaveBeenLastCalledWith('met2', 'met1');

    moveChartDown('met1');
    expect(onReorder).toHaveBeenLastCalledWith('met1', 'met2');

    moveChartDown('met2');
    expect(onReorder).toHaveBeenLastCalledWith('met2', 'met3');
  });
});

describe('useOrderedCharts', () => {
  const mockStorage = new (class {
    private savedData: any = null;
    setItem(key: any, data: any) {
      this.savedData = data;
    }
    getItem() {
      return this.savedData;
    }
    clear() {
      this.savedData = null;
    }
  })();
  jest.spyOn(LocalStorageUtils, 'getStoreForComponent').mockImplementation(() => mockStorage as any);

  const baseMetricKeys = ['met1', 'met2', 'met3', 'met4'];

  beforeEach(() => {
    mockStorage.clear();
  });

  const mountTestComponent = () => {
    const TestComponent = () => {
      const { orderedMetricKeys, onReorderChart } = useOrderedCharts(baseMetricKeys, 'test', 'some-storage-key');

      return (
        <div>
          <div>{orderedMetricKeys.join(',')}</div>
          <button onClick={() => onReorderChart('met1', 'met2')}>move met1 into met2 place</button>
          <button onClick={() => onReorderChart('met1', 'met3')}>move met1 into met3 place</button>
          <button onClick={() => onReorderChart('met3', 'met2')}>move met3 into met2 place</button>
        </div>
      );
    };
    return render(<TestComponent />);
  };

  test('Properly initializes state, reorders charts and persists the state', async () => {
    const { unmount } = mountTestComponent();

    // Expect original, unmodified order
    expect(screen.getByText('met1,met2,met3,met4')).toBeInTheDocument();

    // met1 swaps with met3
    await userEvent.click(screen.getByText('move met1 into met3 place'));
    expect(screen.getByText('met3,met2,met1,met4')).toBeInTheDocument();

    // met3 swaps with met2
    await userEvent.click(screen.getByText('move met3 into met2 place'));
    expect(screen.getByText('met2,met3,met1,met4')).toBeInTheDocument();

    // Unmount the component
    unmount();

    // ...and remount it back
    mountTestComponent();

    // Expect to have the same state as before unmounting
    expect(screen.getByText('met2,met3,met1,met4')).toBeInTheDocument();
  });

  test('Calculates positions when persisted state does not contains all keys', () => {
    mockStorage.setItem('key', JSON.stringify(['met3', 'met1']));
    mountTestComponent();

    // Keep order of known keys and append new ones at the end
    expect(screen.getByText('met3,met1,met2,met4')).toBeInTheDocument();
  });

  test('Calculates positions when persisted state contains unknown keys', () => {
    mockStorage.setItem('key', JSON.stringify(['met5', 'met4', 'met9', 'met1', 'met2', 'met3']));
    mountTestComponent();

    // Keep order of known keys and append new ones at the end
    expect(screen.getByText('met4,met1,met2,met3')).toBeInTheDocument();
  });
});
