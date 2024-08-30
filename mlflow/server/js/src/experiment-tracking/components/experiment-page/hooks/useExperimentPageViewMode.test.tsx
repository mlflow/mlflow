import { act, renderHook } from '@testing-library/react';
import { getExperimentPageDefaultViewMode, useExperimentPageViewMode } from './useExperimentPageViewMode';
import { useLocation } from '../../../../common/utils/RoutingUtils';
import { TestRouter, testRoute } from '../../../../common/utils/RoutingTestUtils';
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

  const mountHook = async (initialPath = '/') => {
    const renderResult = renderHook(() => useExperimentPageViewMode(), {
      wrapper: ({ children }) => (
        <TestRouter
          initialEntries={[initialPath]}
          routes={[
            testRoute(
              <>
                <LocationSpy />
                <div>{children}</div>
              </>,
            ),
          ]}
        />
      ),
    });

    return renderResult;
  };
  test('start with uninitialized state and cycle through modes', async () => {
    const { result } = await mountHook();
    const [, setMode] = result.current;
    expect(result.current[0]).toEqual(getExperimentPageDefaultViewMode());

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
      setMode('TABLE');
    });

    expect(result.current[0]).toEqual('TABLE');
    expect(locationSpyFn).toHaveBeenLastCalledWith('/?compareRunsMode=TABLE');
  });

  test('correctly return preinitialized state', async () => {
    const { result } = await mountHook('/something/?compareRunsMode=ARTIFACT');
    expect(result.current[0]).toEqual('ARTIFACT');
  });
});
