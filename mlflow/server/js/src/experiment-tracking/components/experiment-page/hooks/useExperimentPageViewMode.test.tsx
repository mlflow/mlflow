import { act, renderHook } from '@testing-library/react-hooks';
import { useExperimentPageViewMode } from './useExperimentPageViewMode';
import { MemoryRouter, Routes, Route, useLocation } from '../../../../common/utils/RoutingUtils';
import { useEffect } from 'react';

describe('useExperimentPageViewMode', () => {
  const locationSpyFn = jest.fn();
  const LocationSpy = () => {
    const location = useLocation();
    useEffect(() => {
      locationSpyFn([location.pathname, location.search].join(''));
    }, [location]);
    return null;
  };

  const mountHook = (initialPath = '/') => {
    return renderHook(() => useExperimentPageViewMode(), {
      wrapper: ({ children }) => (
        <MemoryRouter initialEntries={[initialPath]}>
          <LocationSpy />
          <Routes>
            <Route path="*" element={<div>{children}</div>} />
          </Routes>
        </MemoryRouter>
      ),
    });
  };
  test('start with uninitialized state and cycle through modes', () => {
    const { result } = mountHook();
    const [, setMode] = result.current;
    expect(result.current[0]).toEqual(undefined);

    act(() => {
      setMode('ARTIFACT');
    });
    expect(result.current[0]).toEqual('ARTIFACT');
    expect(locationSpyFn).toHaveBeenLastCalledWith('/?compareRunsMode=ARTIFACT');

    act(() => {
      setMode('CHART');
    });
    expect(result.current[0]).toEqual('CHART');
    expect(locationSpyFn).toHaveBeenLastCalledWith('/?compareRunsMode=CHART');

    act(() => {
      setMode(undefined);
    });

    expect(result.current[0]).toEqual(undefined);
    expect(locationSpyFn).toHaveBeenLastCalledWith('/?compareRunsMode=');
  });

  test('correctly return preinitialized state', () => {
    const { result } = mountHook('/something/?compareRunsMode=ARTIFACT');
    expect(result.current[0]).toEqual('ARTIFACT');
  });
});
