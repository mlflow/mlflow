import { describe, test, expect, jest, afterEach } from '@jest/globals';
import LocalStorageUtils, { safeSetItem } from './LocalStorageUtils';

/**
 * Background: in real browsers, `localStorage.setItem` throws a DOMException
 * with name 'QuotaExceededError' when storage is full. JSDOM doesn't enforce
 * quotas, so we cannot reproduce that natively in Jest — instead we spy on
 * `setItem` to throw the same error and assert that LocalStorageStore swallows
 * it (quota errors) or re-throws it (everything else).
 */

afterEach(() => {
  jest.restoreAllMocks();
});

function simulateFullStorage() {
  const throwQuota = () => {
    throw new DOMException(
      "Failed to execute 'setItem' on 'Storage': Setting the value exceeded the quota.",
      'QuotaExceededError',
    );
  };
  jest.spyOn(localStorage, 'setItem').mockImplementation(throwQuota);
  jest.spyOn(sessionStorage, 'setItem').mockImplementation(throwQuota);
}

// The "no spies" describe block must run before "spied" — JSDOM's setItem
// doesn't fully restore after a spy/mockRestore cycle (subsequent calls become
// no-ops). Jest runs describe blocks in declaration order, so this ordering is
// structural rather than positional.
describe('no spies — real localStorage', () => {
  test('Normal operation still works when storage is NOT full', () => {
    const store = LocalStorageUtils.getStoreForComponent('QuotaTest', 'normal');

    store.setItem('key1', 'value1');
    expect(store.getItem('key1')).toEqual('value1');

    store.saveComponentState({ searchInput: 'hello' });
    expect(store.loadComponentState().searchInput).toEqual('hello');
  });
});

describe('spied setItem — quota and error handling', () => {
  test('FIX: LocalStorageStore.setItem does NOT throw when storage is full', () => {
    const store = LocalStorageUtils.getStoreForComponent('ExperimentPage', '["413","414"]');

    simulateFullStorage();

    expect(() => {
      store.setItem(
        'ReactComponentState',
        JSON.stringify({
          compareRunCharts: new Array(3000).fill({
            type: 'LINE',
            metricKey: 'train/loss',
            uuid: 'abc-123',
            isGenerated: true,
            deleted: false,
          }),
        }),
      );
    }).not.toThrow();
  });

  test('FIX: LocalStorageStore.saveComponentState does NOT throw when storage is full', () => {
    const store = LocalStorageUtils.getStoreForComponent('ExperimentPage', '["413","414"]');

    simulateFullStorage();

    expect(() => {
      store.saveComponentState({
        compareRunCharts: new Array(3000).fill({ type: 'BAR', metricKey: 'x' }),
        compareRunSections: new Array(500).fill({ name: 'section', uuid: 'x' }),
      });
    }).not.toThrow();
  });

  test('FIX: sessionStorage setItem also does NOT throw when storage is full', () => {
    const store = LocalStorageUtils.getSessionScopedStoreForComponent('ExperimentPage', '413');

    simulateFullStorage();

    expect(() => {
      store.setItem('chartUIState', 'x'.repeat(10_000_000));
    }).not.toThrow();
  });

  test('Non-quota DOMException errors are re-thrown (not swallowed)', () => {
    jest.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('Access denied', 'SecurityError');
    });

    const store = LocalStorageUtils.getStoreForComponent('ExperimentPage', '["413"]');
    expect(() => store.setItem('anything', 'value')).toThrow('Access denied');
  });

  test('Non-DOMException errors are re-thrown', () => {
    jest.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new TypeError('Cannot read properties of null');
    });

    const store = LocalStorageUtils.getStoreForComponent('ExperimentPage', '["413"]');
    expect(() => store.setItem('anything', 'value')).toThrow(TypeError);
  });

  test('safeSetItem swallows QuotaExceededError', () => {
    simulateFullStorage();
    expect(() => safeSetItem(window.localStorage, 'k', 'v', 'test state')).not.toThrow();
  });

  test('safeSetItem re-throws non-quota DOMException', () => {
    jest.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('Access denied', 'SecurityError');
    });
    expect(() => safeSetItem(window.localStorage, 'k', 'v', 'test state')).toThrow('Access denied');
  });

  test('safeSetItem re-throws non-DOMException', () => {
    jest.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new TypeError('boom');
    });
    expect(() => safeSetItem(window.localStorage, 'k', 'v', 'test state')).toThrow(TypeError);
  });
});
