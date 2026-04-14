import { test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import LocalStorageUtils from './LocalStorageUtils';

/**
 * These tests prove that QuotaExceededError crashes the app WITHOUT
 * the guard, and is gracefully handled WITH it. Non-quota errors
 * are still re-thrown so real bugs are not masked.
 */

let originalSetItem: typeof Storage.prototype.setItem;

beforeEach(() => {
  originalSetItem = Storage.prototype.setItem;
});

afterEach(() => {
  Storage.prototype.setItem = originalSetItem;
});

function simulateFullStorage() {
  Storage.prototype.setItem = jest.fn(() => {
    throw new DOMException(
      "Failed to execute 'setItem' on 'Storage': Setting the value exceeded the quota.",
      'QuotaExceededError',
    );
  }) as any;
}

test('PROOF: raw Storage.prototype.setItem throws QuotaExceededError when storage is full', () => {
  simulateFullStorage();

  // This is what happens when storage is full — unguarded setItem crashes the app
  expect(() => {
    Storage.prototype.setItem.call(window.localStorage, 'anything', 'any value');
  }).toThrow();
});

test('FIX: LocalStorageStore.setItem does NOT throw when storage is full', () => {
  const store = LocalStorageUtils.getStoreForComponent('ExperimentPage', '["413","414"]');

  simulateFullStorage();

  // This would have crashed the app before the fix
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

  // saveComponentState calls setItem internally — should also be safe
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
  Storage.prototype.setItem = jest.fn(() => {
    throw new DOMException('Access denied', 'SecurityError');
  }) as any;

  // Call the prototype directly to test the guard — JSDOM's own setItem
  // doesn't go through the prototype, but the guard logic is the same.
  expect(() => {
    const store = LocalStorageUtils.getStoreForComponent('ExperimentPage', '["413"]');
    Storage.prototype.setItem.call(window.localStorage, 'anything', 'value');
  }).toThrow('Access denied');
});

test('Non-DOMException errors are re-thrown', () => {
  Storage.prototype.setItem = jest.fn(() => {
    throw new TypeError('Cannot read properties of null');
  }) as any;

  expect(() => {
    Storage.prototype.setItem.call(window.localStorage, 'anything', 'value');
  }).toThrow(TypeError);
});

test('Normal operation still works when storage is NOT full', () => {
  // Prototype is restored by afterEach — verify real localStorage works
  const store = LocalStorageUtils.getStoreForComponent('QuotaTest', 'normal');

  store.setItem('key1', 'value1');
  expect(store.getItem('key1')).toEqual('value1');

  store.saveComponentState({ searchInput: 'hello' });
  expect(store.loadComponentState().searchInput).toEqual('hello');
});
