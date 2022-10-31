import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import Utils from '../../../../common/utils/Utils';
import { SearchExperimentRunsFacetsState } from '../models/SearchExperimentRunsFacetsState';
import {
  persistExperimentSearchFacetsState,
  restoreExperimentSearchFacetsState,
} from './persistSearchFacets';

jest.mock('../../../../common/utils/LocalStorageUtils', () => ({
  getStoreForComponent: () => ({
    loadComponentState: jest.fn(),
    saveComponentState: jest.fn(),
  }),
}));

const MOCK_SEARCH_QUERY_PARAMS =
  '?searchFilter=filter%20from%20the%20url&orderByAsc=true&orderByKey=some-key&selectedColumns=persistedCol1,persistedCol2';

describe('persistSearchFacet', () => {
  const mockLocalStorageState = (state: any, saveComponentState: any = jest.fn()) => {
    LocalStorageUtils.getStoreForComponent = jest.fn().mockReturnValue({
      loadComponentState: jest.fn().mockReturnValue(state),
      saveComponentState,
    });
    return saveComponentState;
  };
  beforeEach(() => {
    mockLocalStorageState({});
  });

  describe('restoreExperimentSearchFacetsState', () => {
    test('it should properly extract simple URL', () => {
      const { state } = restoreExperimentSearchFacetsState(MOCK_SEARCH_QUERY_PARAMS, 'id-key');
      expect(state.searchFilter).toEqual('filter from the url');
      expect(state.orderByAsc).toEqual(true);
      expect(state.orderByKey).toEqual('some-key');
      expect(state.selectedColumns).toEqual(['persistedCol1', 'persistedCol2']);
    });

    test('it properly parses single- and double-encoded search params', () => {
      expect(
        restoreExperimentSearchFacetsState(
          `orderByKey=${encodeURIComponent('params.`param_1`')}`,
          'id-key',
        ).state.orderByKey,
      ).toEqual('params.`param_1`');

      expect(
        restoreExperimentSearchFacetsState(
          `orderByKey=${encodeURIComponent(encodeURIComponent('params.`param_1`'))}`,
          'id-key',
        ).state.orderByKey,
      ).toEqual('params.`param_1`');
    });

    test('it should properly extract simple state from local storage', () => {
      mockLocalStorageState({
        searchFilter: 'from-local-storage',
        orderByAsc: true,
        orderByKey: 'some-local-storage-sort-key',
      });
      const { state } = restoreExperimentSearchFacetsState('', 'id-key');
      expect(state.searchFilter).toEqual('from-local-storage');
      expect(state.orderByAsc).toEqual(true);
      expect(state.orderByKey).toEqual('some-local-storage-sort-key');
    });

    test('it should properly merge URL state with local storage state', () => {
      mockLocalStorageState({
        searchFilter: 'from-local-storage',
        orderByAsc: true,
        orderByKey: 'some-local-storage-sort-key',
      });
      const { state } = restoreExperimentSearchFacetsState('?searchFilter=urlfilter', 'id-key');
      expect(state.searchFilter).toEqual('urlfilter');

      // URL state (complemented with default values) should overshadow settings from the storage
      expect(state.orderByAsc).not.toEqual(true);
      expect(state.orderByKey).not.toEqual('some-local-storage-sort-key');
    });

    test('it should properly re-persist the local storage after merging', () => {
      const saveState = mockLocalStorageState({
        searchFilter: 'from-local-storage',
        orderByAsc: true,
        orderByKey: 'some-local-storage-sort-key',
      });
      restoreExperimentSearchFacetsState(MOCK_SEARCH_QUERY_PARAMS, 'id-key');
      expect(saveState).toBeCalledWith(
        expect.objectContaining({
          searchFilter: 'filter from the url',
        }),
      );
    });

    test('it should properly react to faulty data', () => {
      Utils.logErrorAndNotifyUser = jest.fn();
      LocalStorageUtils.getStoreForComponent = jest.fn().mockReturnValue({
        loadComponentState: jest.fn().mockImplementation(() => {
          throw new Error();
        }),
        saveComponentState: jest.fn(),
      });

      const state = restoreExperimentSearchFacetsState(MOCK_SEARCH_QUERY_PARAMS, 'id-key');
      expect(state).toBeTruthy();
      expect(Utils.logErrorAndNotifyUser).toBeCalledTimes(1);
    });

    test('it marks the calculated state as pristine if no changes are done', () => {
      const { isPristine } = restoreExperimentSearchFacetsState('', 'id-key');
      expect(isPristine).toEqual(true);
    });
  });

  describe('persistExperimentSearchFacetsState', () => {
    test('it persist simple state to the store', () => {
      const saveStateFn = mockLocalStorageState({});
      const state = new SearchExperimentRunsFacetsState();
      state.searchFilter = 'some filter';
      state.orderByKey = 'order-key';
      state.orderByAsc = true;
      state.selectedColumns = ['col1', 'col2'];

      persistExperimentSearchFacetsState(state, 'id-key');

      expect(saveStateFn).toBeCalledWith(
        expect.objectContaining({
          searchFilter: 'some filter',
          orderByKey: 'order-key',
          orderByAsc: true,
          selectedColumns: ['col1', 'col2'],
        }),
      );
    });

    test('it persist simple state to the URL', () => {
      const state = new SearchExperimentRunsFacetsState();
      state.searchFilter = 'some filter';
      state.orderByKey = 'order-key';
      state.orderByAsc = true;
      state.selectedColumns = ['col1', 'col2'];

      const queryString = persistExperimentSearchFacetsState(state, 'id-key');

      expect(queryString).toEqual(
        '?searchFilter=some%20filter&orderByKey=order-key&orderByAsc=true&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All%20Runs&selectedColumns=col1,col2',
      );
    });

    test('it persist state with existing query param to the URL', () => {
      const state = new SearchExperimentRunsFacetsState();
      state.searchFilter = 'some filter';
      state.orderByKey = 'order-key';
      state.orderByAsc = true;

      const queryString = persistExperimentSearchFacetsState(
        state,
        'id-key',
        '?experiments=foobar&somethingElse=abc',
      );

      let expectedQuery = '?experiments=foobar';
      expectedQuery += `&searchFilter=${encodeURIComponent('some filter')}`;
      expectedQuery += `&orderByKey=${encodeURIComponent('order-key')}`;
      expectedQuery += `&orderByAsc=${encodeURIComponent('true')}`;
      expectedQuery += `&startTime=${encodeURIComponent('ALL')}`;
      expectedQuery += `&lifecycleFilter=${encodeURIComponent('Active')}`;
      expectedQuery += `&modelVersionFilter=${encodeURIComponent('All Runs')}`;
      expectedQuery += `&selectedColumns=${state.selectedColumns
        .map((c) => encodeURIComponent(c))
        .join(',')}`;

      expect(queryString).toEqual(expectedQuery);
    });
  });
});
