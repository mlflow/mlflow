import { act, renderHook } from '@testing-library/react';
import { useExperimentPageSearchFacets, useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { MemoryRouter, Routes, Route, useLocation, useSearchParams } from '../../../../common/utils/RoutingUtils';
import { testRoute, TestRouter } from '../../../../common/utils/RoutingTestUtils';
import { useEffect } from 'react';
import { screen, renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';

describe('useExperimentPageSearchFacets', () => {
  const mountHook = async (initialPath = '/') => {
    const hookResult = renderHook(() => useExperimentPageSearchFacets(), {
      wrapper: ({ children }) => (
        <TestRouter
          initialEntries={[initialPath]}
          routes={[
            testRoute(<div>{children}</div>, '/experiments/:experimentId'),
            testRoute(<div>{children}</div>, '/compare-experiments'),
          ]}
        />
      ),
    });

    return hookResult;
  };
  test('return null for uninitialized state', async () => {
    const { result } = await mountHook('/experiments/123');
    expect(result.current).toEqual([null, ['123'], false]);
  });

  test('return correct data for initialized state', async () => {
    const { result } = await mountHook(
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
      false,
    ]);
  });

  test('return correct data for multiple compared experiments', async () => {
    const { result } = await mountHook('/compare-experiments?experiments=%5B%22444%22%2C%22555%22%5D&orderByKey=foo');
    expect(result.current).toEqual([
      expect.objectContaining({
        orderByKey: 'foo',
      }),
      ['444', '555'],
      false,
    ]);
  });

  test('return empty list when facing invalid compare experiment IDs', async () => {
    const { result } = await mountHook('/compare-experiments?experiments=__invalid_array__&orderByKey=foo');
    expect(result.current).toEqual([
      expect.objectContaining({
        orderByKey: 'foo',
      }),
      [
        /* empty */
      ],
      false,
    ]);
  });

  test('ignore unrelated parameters', async () => {
    const { result } = await mountHook('/experiments/123?orderByKey=foo&o=123456');
    expect(result.current).toEqual([
      expect.objectContaining({
        orderByKey: 'foo',
      }),
      ['123'],
      false,
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
    const { result } = await mountHook('/experiments/123?orderByKey=foo&o=123456');

    expect(result.current).toEqual([
      {
        ...createExperimentPageSearchFacetsState(),
        orderByKey: 'foo',
      },
      ['123'],
      false,
    ]);
  });

  test('reports if the view is in preview mode', async () => {
    const { result } = await mountHook('/experiments/123?o=123456&isPreview=true');

    expect(result.current).toEqual([null, ['123'], true]);
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

  const mountHook = async (initialPath = '/') => {
    const hookResult = renderHook(() => useUpdateExperimentPageSearchFacets(), {
      wrapper: ({ children }) => (
        <>
          <TestRouter
            initialEntries={[initialPath]}
            routes={[
              testRoute(
                <div>
                  <LocationSpy />
                  {children}
                </div>,
                '/experiments/:experimentId',
              ),
              testRoute(
                <div>
                  <LocationSpy />
                  {children}
                </div>,
                '/compare-experiments',
              ),
            ]}
          />
        </>
      ),
    });

    return hookResult;
  };

  test('correctly change search facets for single experiment', async () => {
    const { result } = await mountHook('/experiments/123');
    let updateFn = result.current;
    act(() => {
      updateFn({ orderByKey: 'some-column', orderByAsc: true });
    });
    expect(locationSpyFn).toHaveBeenLastCalledWith('/experiments/123?orderByKey=some-column&orderByAsc=true');

    updateFn = result.current;
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
    const { result } = await mountHook('/compare-experiments?experiments=%5B%22444%22%2C%22555%22%5D');

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
    const { result } = await mountHook('/experiments/123?o=12345');
    const updateFn = result.current;
    act(() => {
      updateFn({ orderByKey: 'some-column' });
    });
    expect(locationSpyFn).toHaveBeenLastCalledWith('/experiments/123?o=12345&orderByKey=some-column');
  });

  test('correctly disable preview mode when params are explicitly changed', async () => {
    const { result } = await mountHook('/experiments/123?o=12345&&orderByKey=abc&isPreview=true');
    const updateFn = result.current;
    act(() => {
      updateFn({ orderByKey: 'some-column' });
    });
    expect(locationSpyFn).toHaveBeenLastCalledWith('/experiments/123?o=12345&orderByKey=some-column');
  });
});
