import { shouldEnableShareExperimentViewByTags } from '../../../../../common/utils/FeatureUtils';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { renderWithIntl, act, screen } from 'common/utils/TestUtils.react17';
import { searchRunsPayload } from '../../../../actions';
import { POLL_INTERVAL } from '../../../../constants';
import { ExperimentViewRefreshButtonImpl } from './ExperimentViewRefreshButton';

const mockRunsContext = {
  actions: { searchRunsPayload: jest.fn().mockResolvedValue({}) },
  updateSearchFacets: jest.fn(),
};

jest.mock('../../../../../common/utils/FeatureUtils', () => ({
  shouldEnableShareExperimentViewByTags: jest.fn().mockReturnValue(false),
}));

jest.mock('../../../../actions', () => ({
  searchRunsPayload: jest.fn().mockResolvedValue({ type: 'mockedAction' }),
}));

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

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
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

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
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

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
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

  // TODO: Remove/migrate to RTL once we fully migrate to the new view state model
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

  describe('using new view state model', () => {
    beforeEach(() => {
      jest.mocked(shouldEnableShareExperimentViewByTags).mockReturnValue(true);
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
