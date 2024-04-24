import { act, renderHook } from '@testing-library/react-for-react-18';
import { useExperimentPageSearchFacets, useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import {
  MemoryRouter,
  Routes,
  Route,
  useLocation,
  useNavigate,
  useSearchParams,
} from '../../../../common/utils/RoutingUtils';
import { useEffect } from 'react';
import { screen, renderWithIntl } from 'common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event-14';
import { createExperimentPageSearchFacetsStateV2 } from '../models/ExperimentPageSearchFacetsStateV2';

describe('useExperimentPageSearchFacets', () => {
  const mountHook = (initialPath = '/') =>
    renderHook(() => useExperimentPageSearchFacets(), {
      wrapper: ({ children }) => (
        <MemoryRouter initialEntries={[initialPath]}>
          <Routes>
            <Route path="/experiments/:experimentId" element={<div>{children}</div>} />
            <Route path="/compare-experiments" element={<div>{children}</div>} />
          </Routes>
        </MemoryRouter>
      ),
    });
  test('return null for uninitialized state', () => {
    const { result } = mountHook('/experiments/123');
    expect(result.current).toEqual([null, ['123']]);
  });

  test('return correct data for initialized state', () => {
    const { result } = mountHook(
      '/experiments/123?orderByKey=foo&orderByAsc=true&searchFilter=test%20string&datasetsFilter=W10=&lifecycleFilter=ACTIVE&modelVersionFilter=All Runs&startTime=ALL',
    );
    expect(result.current).toEqual([
      {
        orderByAsc: true,
        orderByKey: 'foo',
        searchFilter: 'test string',
        datasetsFilter: [],
        lifecycleFilter: 'ACTIVE',
        modelVersionFilter: 'All Runs',
        startTime: 'ALL',
      },
      ['123'],
    ]);
  });

  test('return correct data for multiple compared experiments', () => {
    const { result } = mountHook('/compare-experiments?experiments=%5B%22444%22%2C%22555%22%5D&orderByKey=foo');
    expect(result.current).toEqual([
      expect.objectContaining({
        orderByKey: 'foo',
      }),
      ['444', '555'],
    ]);
  });

  test('return empty list when facing invalid compare experiment IDs', () => {
    const { result } = mountHook('/compare-experiments?experiments=__invalid_array__&orderByKey=foo');
    expect(result.current).toEqual([
      expect.objectContaining({
        orderByKey: 'foo',
      }),
      [
        /* empty */
      ],
    ]);
  });

  test('ignore unrelated parameters', () => {
    const { result } = mountHook('/experiments/123?orderByKey=foo&o=123456');
    expect(result.current).toEqual([
      expect.objectContaining({
        orderByKey: 'foo',
      }),
      ['123'],
    ]);
  });

  test('return the same reference when unrelated query params change', async () => {
    const changeFacetsSpy = jest.fn();
    const TestComponent = () => {
      const [, changeParams] = useSearchParams();
      const [searchFacets] = useExperimentPageSearchFacets();
      useEffect(() => {
        changeFacetsSpy();
      }, [searchFacets]);
      return (
        <div>
          <button
            onClick={() =>
              changeParams((params) => {
                params.set('orderByKey', 'foobar');
                return params;
              })
            }
          >
            Change order
          </button>
          <button
            onClick={() =>
              changeParams((params) => {
                params.set('somethingelse', 'xyz');
                return params;
              })
            }
          >
            Change other parameter
          </button>
        </div>
      );
    };
    renderWithIntl(
      <MemoryRouter initialEntries={['/experiments/123?orderByKey=abc']}>
        <Routes>
          <Route path="/experiments/:experimentId" element={<TestComponent />} />
        </Routes>
      </MemoryRouter>,
    );

    // Initial render
    expect(changeFacetsSpy).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByText('Change order'));

    // Should trigger re-render because of change in orderByKey
    expect(changeFacetsSpy).toHaveBeenCalledTimes(2);

    await userEvent.click(screen.getByText('Change other parameter'));

    // Should not trigger re-render because of change in unrelated search param
    expect(changeFacetsSpy).toHaveBeenCalledTimes(2);
  });

  test('fills gaps when only partial facets are provided', async () => {
    const { result } = mountHook('/experiments/123?orderByKey=foo&o=123456');

    expect(result.current).toEqual([
      {
        ...createExperimentPageSearchFacetsStateV2(),
        orderByKey: 'foo',
      },
      ['123'],
    ]);
  });
});

describe('useUpdateExperimentPageSearchFacets', () => {
  const locationSpyFn = jest.fn();
  const LocationSpy = () => {
    const location = useLocation();
    useEffect(() => {
      locationSpyFn([location.pathname, location.search].join(''));
    }, [location]);
    return null;
  };
  const mountHook = (initialPath = '/') => {
    return renderHook(() => useUpdateExperimentPageSearchFacets(), {
      wrapper: ({ children }) => (
        <MemoryRouter initialEntries={[initialPath]}>
          <LocationSpy />
          <Routes>
            <Route path="/experiments/:experimentId" element={<div>{children}</div>} />
            <Route path="/compare-experiments" element={<div>{children}</div>} />
          </Routes>
        </MemoryRouter>
      ),
    });
  };

  test('correctly change search facets for single experiment', async () => {
    const { result } = mountHook('/experiments/123');
    const updateFn = result.current;
    act(() => {
      updateFn({ orderByKey: 'some-column', orderByAsc: true });
    });
    expect(locationSpyFn).toHaveBeenLastCalledWith('/experiments/123?orderByKey=some-column&orderByAsc=true');
    act(() => {
      updateFn({
        datasetsFilter: [{ context: 'ctx', digest: 'digest', experiment_id: '123', name: 'name' }],
      });
    });
    expect(locationSpyFn).toHaveBeenLastCalledWith(
      '/experiments/123?orderByKey=some-column&orderByAsc=true&datasetsFilter=W3sibmFtZSI6Im5hbWUiLCJkaWdlc3QiOiJkaWdlc3QiLCJjb250ZXh0IjoiY3R4In1d',
    );
  });

  test('correctly change search facets for compare experiments', async () => {
    const { result } = mountHook('/compare-experiments?experiments=%5B%22444%22%2C%22555%22%5D');

    expect(locationSpyFn).toHaveBeenLastCalledWith('/compare-experiments?experiments=%5B%22444%22%2C%22555%22%5D');

    const updateFn = result.current;
    act(() => {
      updateFn({ orderByKey: 'some-column' });
    });
    expect(locationSpyFn).toHaveBeenLastCalledWith(
      '/compare-experiments?experiments=%5B%22444%22%2C%22555%22%5D&orderByKey=some-column',
    );
  });

  test('correctly retain unrelated parameters', async () => {
    const { result } = mountHook('/experiments/123?o=12345');
    const updateFn = result.current;
    act(() => {
      updateFn({ orderByKey: 'some-column' });
    });
    expect(locationSpyFn).toHaveBeenLastCalledWith('/experiments/123?o=12345&orderByKey=some-column');
  });
});
