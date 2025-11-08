import type { NotificationInstance } from '@databricks/design-system';
import { useEffect } from 'react';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import type { RunEntity, RunInfoEntity } from '../../../types';
import { EXPERIMENT_PARENT_ID_TAG } from '../utils/experimentPage.common-utils';
import { useFetchedRunsNotification } from './useFetchedRunsNotification';

const generateRuns = (n: number, asChildRuns = false): RunEntity[] =>
  new Array(n).fill(0).map(
    (_, index) =>
      ({
        info: { runUuid: asChildRuns ? `run_child${index}` : `run${index}` },
        data: asChildRuns ? { tags: [{ key: EXPERIMENT_PARENT_ID_TAG, value: `parent-id-${index}` }] } : undefined,
      } as any),
  );

describe('useFetchedRunsNotification', () => {
  const notificationInstance = {
    close: jest.fn(),
    info: jest.fn(),
  } as any as NotificationInstance;

  const createWrapper = (fetchedRuns: RunEntity[], existingRunInfos: RunInfoEntity[] = []) => {
    const TestComponent = () => {
      const showNotification = useFetchedRunsNotification(notificationInstance);
      useEffect(() => {
        showNotification(fetchedRuns, existingRunInfos);
      }, [showNotification]);
      return <div />;
    };
    return mountWithIntl(<TestComponent />);
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('displays proper notification for mixed runs', () => {
    createWrapper([...generateRuns(7, false), ...generateRuns(3, true)]);

    expect(notificationInstance.info).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Loaded 10 runs, including 3 child runs' }),
    );
  });

  it('displays proper notification for mixed runs w/ correct pluralization', () => {
    createWrapper([...generateRuns(9, false), ...generateRuns(1, true)]);

    expect(notificationInstance.info).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Loaded 10 runs, including 1 child run' }),
    );
  });

  it('displays proper notification for child-only runs', () => {
    createWrapper(generateRuns(10, true));

    expect(notificationInstance.info).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Loaded 10 child runs' }),
    );
  });

  it('displays notification with runs properly counted while excluding existing runs', () => {
    const fetchedRuns = [...generateRuns(50, false), ...generateRuns(50, true)];

    const existingRunInfos = [
      ...fetchedRuns.slice(0, 10).map(({ info }) => info), // 10 existing parent runs
      ...fetchedRuns.slice(50, 80).map(({ info }) => info), // 30 existing child runs
    ];
    createWrapper(fetchedRuns, existingRunInfos);

    expect(notificationInstance.info).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Loaded 60 runs, including 20 child runs' }),
    );
  });

  it('does not display notification when no runs are fetched', () => {
    createWrapper([]);
    expect(notificationInstance.info).not.toHaveBeenCalled();
  });
});
