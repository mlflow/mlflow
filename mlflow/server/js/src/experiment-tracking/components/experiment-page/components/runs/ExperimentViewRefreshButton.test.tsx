import { mountWithIntl } from '../../../../../common/utils/TestUtils';
import { POLL_INTERVAL } from '../../../../constants';
import { ExperimentViewRefreshButtonImpl } from './ExperimentViewRefreshButton';

const mockRunsContext = {
  actions: { searchRunsPayload: jest.fn().mockResolvedValue({}) },
  updateSearchFacets: jest.fn(),
};

jest.mock('../../hooks/useFetchExperimentRuns', () => ({
  useFetchExperimentRuns: () => mockRunsContext,
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

  test('it should change the number on the badge correctly', () => {
    mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);

    // Wait a half of necessary interval...
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // No calls expected
    expect(mockRunsContext.actions.searchRunsPayload).toBeCalledTimes(0);

    // Wait another half
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // One call expected
    expect(mockRunsContext.actions.searchRunsPayload).toBeCalledTimes(1);
  });

  test('it should reset the interval correctly', () => {
    const wrapper = mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);

    // Wait a half of necessary interval...
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // No calls expected
    expect(mockRunsContext.actions.searchRunsPayload).toBeCalledTimes(0);

    // In the meanwhile, update run infos in the "store"
    wrapper.setProps({ runInfos: {} } as any);

    // Wait another half
    jest.advanceTimersByTime(POLL_INTERVAL / 2);

    // Still no calls expected
    expect(mockRunsContext.actions.searchRunsPayload).toBeCalledTimes(0);
  });

  test('does the correct call for new runs', () => {
    const dateNowSpy = jest.spyOn(Date, 'now').mockImplementation(() => 1000);

    mockRunsContext.actions.searchRunsPayload.mockResolvedValue({
      runs: [{}, {}, {}, {}],
    });
    mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);

    // Wait for the interval time
    jest.advanceTimersByTime(POLL_INTERVAL);

    // One call expected
    expect(mockRunsContext.actions.searchRunsPayload).toBeCalledTimes(1);
    expect(mockRunsContext.actions.searchRunsPayload).toBeCalledWith(
      expect.objectContaining({
        experimentIds: [1, 2, 3],
        filter: 'attributes.start_time > 1000',
      }),
    );

    dateNowSpy.mockRestore();
  });

  test('refreshes the runs when clicked', () => {
    const wrapper = mountWithIntl(<ExperimentViewRefreshButtonImpl runInfos={{}} />);
    wrapper.find('button').simulate('click');
    expect(mockRunsContext.updateSearchFacets).toBeCalledWith(
      {},
      {
        forceRefresh: true,
        preservePristine: true,
      },
    );
  });
});
