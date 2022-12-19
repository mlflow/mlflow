import { shouldUseNextRunsComparisonUI } from '../../../../common/utils/FeatureUtils';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import Utils from '../../../../common/utils/Utils';
import { SearchExperimentRunsFacetsState } from '../models/SearchExperimentRunsFacetsState';
import {
  persistExperimentSearchFacetsState,
  restoreExperimentSearchFacetsState,
} from './persistSearchFacets';
import {
  deserializeFieldsFromLocalStorage,
  deserializeFieldsFromQueryString,
  serializeFieldsToLocalStorage,
  serializeFieldsToQueryString,
} from './persistSearchFacets.serializers';

jest.mock('./persistSearchFacets.serializers.ts', () => ({
  serializeFieldsToQueryString: jest.fn(),
  serializeFieldsToLocalStorage: jest.fn(),
  deserializeFieldsFromLocalStorage: jest.fn(),
  deserializeFieldsFromQueryString: jest.fn(),
}));

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  shouldUseNextRunsComparisonUI: jest.fn(),
}));

jest.mock('../../../../common/utils/LocalStorageUtils', () => ({
  getStoreForComponent: () => ({
    loadComponentState: jest.fn(),
    saveComponentState: jest.fn(),
  }),
}));

const MOCK_SEARCH_QUERY_PARAMS =
  '?searchFilter=filter%20from%20the%20url&orderByAsc=true&orderByKey=some-key&selectedColumns=persistedCol1,persistedCol2';

beforeEach(() => {
  (serializeFieldsToQueryString as jest.Mock).mockImplementation((identity) => identity);
  (serializeFieldsToLocalStorage as jest.Mock).mockImplementation((identity) => identity);
  (deserializeFieldsFromLocalStorage as jest.Mock).mockImplementation((identity) => identity);
  (deserializeFieldsFromQueryString as jest.Mock).mockImplementation((identity) => identity);
  (shouldUseNextRunsComparisonUI as jest.Mock).mockReturnValue(false);
});

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
        '?searchFilter=some%20filter&orderByKey=order-key&orderByAsc=true&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All%20Runs&selectedColumns=col1,col2&isComparingRuns=false',
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
      expectedQuery += `&isComparingRuns=false`;

      expect(queryString).toEqual(expectedQuery);
    });

    test('it persists state using custom field serializers', () => {
      const saveLocalStorageState = mockLocalStorageState({});

      (shouldUseNextRunsComparisonUI as jest.Mock).mockReturnValue(true);
      (serializeFieldsToQueryString as jest.Mock).mockImplementation((input) => {
        return {
          ...input,
          selectedColumns: `qs-serialized-${input.selectedColumns.length}-columns`,
        };
      });
      (serializeFieldsToLocalStorage as jest.Mock).mockImplementation((input) => ({
        ...input,
        selectedColumns: `ls-serialized-${input.selectedColumns.length}-columns`,
      }));

      const persistedQueryString = persistExperimentSearchFacetsState(
        Object.assign(new SearchExperimentRunsFacetsState(), {
          orderByKey: 'some_column',
          selectedColumns: ['column', 'some-column', 'another-column'],
        }),
        'some-id-key',
      );

      expect(persistedQueryString).toContain('orderByKey=some_column');
      expect(persistedQueryString).toContain('selectedColumns=qs-serialized-3-columns');

      expect(saveLocalStorageState).toBeCalledWith(
        expect.objectContaining({
          selectedColumns: 'ls-serialized-3-columns',
        }),
      );
    });

    test('it restores the state using custom field deserializers', () => {
      (shouldUseNextRunsComparisonUI as jest.Mock).mockReturnValue(true);
      (deserializeFieldsFromQueryString as jest.Mock).mockImplementation((input) => {
        return {
          ...input,
          runsPinned: ['deserialized-run-id-qs'],
        };
      });

      const { state } = restoreExperimentSearchFacetsState(
        '?runsPinned=some-serialized-value&orderByKey=some-key',
        'some-id',
      );

      expect(state).toEqual(
        expect.objectContaining({
          runsPinned: ['deserialized-run-id-qs'],
        }),
      );
    });
  });
});
