import { describe, expect, test } from '@jest/globals';
import { act, render } from '@testing-library/react';
// eslint-disable-next-line no-restricted-imports
import { createMemoryRouter } from 'react-router-dom';
import {
  MemoryRouter,
  RouterProvider,
  useLocation,
  useNavigate,
  useSearchParamSelector,
  type Location,
  type NavigateFunction,
} from './RoutingUtils';

const mount = (kind: 'data' | 'memory') => {
  let selected: ReturnType<typeof useSearchParamSelector<string | null>>;
  let location: Location;
  let navigate: NavigateFunction;
  let renders = 0;
  const Consumer = () => {
    selected = useSearchParamSelector((params) => params.get('q'));
    renders++;
    return null;
  };
  const Probe = () => {
    location = useLocation();
    navigate = useNavigate();
    return null;
  };
  const content = (
    <>
      <Consumer />
      <Probe />
    </>
  );
  const initialEntries = [
    {
      pathname: '/base/p',
      search: '?q=first&tag=one&tag=two&filters=%5B%5D',
      hash: '#section',
      state: { retained: true },
    },
  ];
  if (kind === 'data') {
    const router = createMemoryRouter([{ path: '*', element: content }], { basename: '/base', initialEntries });
    render(<RouterProvider router={router} />);
  } else {
    render(
      <MemoryRouter basename="/base" initialEntries={initialEntries}>
        {content}
      </MemoryRouter>,
    );
  }
  return {
    get selected() {
      return selected!;
    },
    get location() {
      return location!;
    },
    get renders() {
      return renders;
    },
    navigate: (to: number) => navigate!(to),
  };
};

describe.each(['data', 'memory'] as const)('%s router selected query params', (kind) => {
  test('a retained setter merges latest writes and preserves basename, hash, state and repeated params', async () => {
    const page = mount(kind);
    const setParams = page.selected[1];
    await act(async () => {
      setParams((params) => {
        params.set('spanId', 'child');
        return params;
      });
      setParams((params) => {
        params.set('q', 'second');
        return params;
      });
    });
    expect(page.selected[0]).toBe('second');
    expect(page.location.pathname).toBe('/p');
    expect(page.location.hash).toBe('#section');
    expect(page.location.state).toEqual({ retained: true });
    const params = new URLSearchParams(page.location.search);
    expect(params.get('spanId')).toBe('child');
    expect(params.getAll('tag')).toEqual(['one', 'two']);
    expect(params.get('filters')).toBe('[]');
    await act(async () => {
      setParams((current) => {
        current.set('page', '2');
        return current;
      });
    });
    expect(new URLSearchParams(page.location.search).get('q')).toBe('second');
  });

  test('replace keeps the current history entry and native history restores the selected value', async () => {
    const page = mount(kind);
    const setParams = page.selected[1];
    await act(async () => {
      setParams({ q: 'second' });
    });
    await act(async () => {
      setParams({ q: 'replacement' }, { replace: true, state: { replaced: true } });
    });
    expect(page.selected[0]).toBe('replacement');
    expect(page.location.state).toEqual({ replaced: true });
    await act(async () => {
      page.navigate(-1);
    });
    expect(page.selected[0]).toBe('first');
    await act(async () => {
      page.navigate(1);
    });
    expect(page.selected[0]).toBe('replacement');
  });

  test('only data-router consumers skip unrelated URL updates', async () => {
    const page = mount(kind);
    const before = page.renders;
    await act(async () => {
      page.selected[1]((params) => {
        params.set('spanId', 'child');
        return params;
      });
    });
    expect(page.selected[0]).toBe('first');
    expect(page.renders - before).toBe(kind === 'data' ? 0 : 1);
  });
});
