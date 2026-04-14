import { test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import LocalStorageUtils from './LocalStorageUtils';

/**
 * These tests prove that QuotaExceededError crashes the app WITHOUT
 * the try/catch guard, and is silently handled WITH it.
 */

let originalSetItem: typeof Storage.prototype.setItem;

beforeEach(() => {
  originalSetItem = Storage.prototype.setItem;
});

afterEach(() => {
  Storage.prototype.setItem = originalSetItem;
});

function simulateFullStorage() {
  // Replace setItem with one that always throws QuotaExceededError
  // This simulates what Chrome does when localStorage is full (~5MB)
  Storage.prototype.setItem = jest.fn((_key: string, _value: string) => {
    const error = new DOMException(
      "Failed to execute 'setItem' on 'Storage': Setting the value exceeded the quota.",
      'QuotaExceededError',
    );
    throw error;
  }) as any;
}

test('PROOF: raw Storage.prototype.setItem throws QuotaExceededError when storage is full', () => {
  simulateFullStorage();

  // This is what happens in vanilla MLflow 3.10 — unguarded setItem via prototype
  // (our LocalStorageStore calls this.storageObj.setItem which goes through the prototype)
  expect(() => {
    Storage.prototype.setItem.call(window.localStorage, 'anything', 'any value');
  }).toThrow();
});

test('FIX: LocalStorageStore.setItem does NOT throw when storage is full', () => {
  const store = LocalStorageUtils.getStoreForComponent('ExperimentPage', '["413","414"]');

  simulateFullStorage();

  // This would have crashed the app before the fix
  expect(() => {
    store.setItem('ReactComponentState', JSON.stringify({
      compareRunCharts: new Array(3000).fill({
        type: 'LINE',
        metricKey: 'train/loss',
        uuid: 'abc-123',
        isGenerated: true,
        deleted: false,
      }),
    }));
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

test('Normal operation still works when storage is NOT full', () => {
  // Don't simulate full storage — use real localStorage
  const store = LocalStorageUtils.getStoreForComponent('QuotaTest', 'normal');

  store.setItem('key1', 'value1');
  expect(store.getItem('key1')).toEqual('value1');

  store.saveComponentState({ searchInput: 'hello' });
  expect(store.loadComponentState().searchInput).toEqual('hello');
});
