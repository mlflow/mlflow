import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { renderWithIntl, act, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react17';
import { searchRunsPayload } from '../../../../actions';
import { POLL_INTERVAL } from '../../../../constants';
import { ExperimentViewRefreshButtonImpl } from './ExperimentViewRefreshButton';

jest.mock('../../../../actions', () => ({
  searchRunsPayload: jest.fn().mockResolvedValue({ type: 'mockedAction' }),
}));

jest.mock('../../hooks/useExperimentIds', () => ({
  useExperimentIds: jest.fn().mockReturnValue([1, 2, 3]),
}));

describe('ExperimentViewRefreshButton', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllTimers();
    jest.clearAllMocks();
  });

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
  test('it should change the number on the badge correctly', () => {
    mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);

    // Wait a half of necessary interval...
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // No calls expected
    expect(searchRunsPayload).toBeCalledTimes(0);

    // Wait another half
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // One call expected
    expect(searchRunsPayload).toBeCalledTimes(1);
  });

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
  test('it should reset the interval correctly', () => {
    const wrapper = mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);

    // Wait a half of necessary interval...
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // No calls expected
    expect(searchRunsPayload).toBeCalledTimes(0);

    // In the meanwhile, update run infos in the "store"
    wrapper.setProps({ runInfos: {} } as any);

    // Wait another half
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // Still no calls expected
    expect(searchRunsPayload).toBeCalledTimes(0);
  });

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
  test('does the correct call for new runs', () => {
    const dateNowSpy = jest.spyOn(Date, 'now').mockImplementation(() => 1000);

    jest.mocked(searchRunsPayload).mockResolvedValue({
      runs: [{}, {}, {}, {}],
    });
    mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);

    // Wait for the interval time
    jest.advanceTimersByTime(POLL_INTERVAL);

    // One call expected
    expect(searchRunsPayload).toBeCalledTimes(1);
    expect(searchRunsPayload).toBeCalledWith(
      expect.objectContaining({
        experimentIds: [1, 2, 3],
        filter: 'attributes.start_time > 1000',
      }),
    );

    dateNowSpy.mockRestore();
  });

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
  test('refreshes the runs when clicked', () => {
    const refreshRuns = jest.fn();
    const wrapper = mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} refreshRuns={refreshRuns} />);
    expect(refreshRuns).not.toBeCalled();
    wrapper.find('button').simulate('click');
    expect(refreshRuns).toBeCalled();
  });

  describe('using new view state model', () => {
    beforeEach(() => {
      jest.mocked(searchRunsPayload).mockClear();
    });

    test('it should call action directly when using new view state model', async () => {
      renderWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);

      await act(async () => {
        // Wait a half of necessary interval...
        jest.advanceTimersByTime(POLL_INTERVAL / 2);
      });

      // No calls expected
      expect(searchRunsPayload).toBeCalledTimes(0);

      await act(async () => {
        // Wait another half
        jest.advanceTimersByTime(POLL_INTERVAL / 2);
      });

      // One call expected
      expect(searchRunsPayload).toBeCalledTimes(1);
    });

    test('it should call provided refresh function when clicked', () => {
      const refreshRuns = jest.fn();
      renderWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} refreshRuns={refreshRuns} />);
      screen.getByRole('button').click();
      expect(refreshRuns).toBeCalled();
    });
  });
});
