import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { IndexedDBStorage } from './IndexedDBStorageUtils';

// Mock IndexedDB with proper async behavior
const createMockIndexedDB = () => {
  const mockStore = {
    put: jest.fn(),
    delete: jest.fn(),
    openCursor: jest.fn(() => ({
      onsuccess: null,
      onerror: null,
      result: null,
    })),
  };

  const mockTransaction = {
    objectStore: jest.fn(() => mockStore),
    oncomplete: null as any,
    onerror: null as any,
    onabort: null as any,
  };

  const mockDB = {
    objectStoreNames: { contains: jest.fn(() => false) },
    createObjectStore: jest.fn(),
    transaction: jest.fn(() => {
      setTimeout(() => {
        if (mockTransaction.oncomplete) mockTransaction.oncomplete();
      }, 0);
      return mockTransaction;
    }),
  };

  global.indexedDB = {
    open: jest.fn(() => {
      const request = {
        result: mockDB,
        onupgradeneeded: null as any,
        onsuccess: null as any,
        onerror: null as any,
      };
      setTimeout(() => {
        if (request.onupgradeneeded) request.onupgradeneeded();
        if (request.onsuccess) request.onsuccess();
      }, 0);
      return request;
    }),
  } as any;
};

describe('IndexedDBStorage', () => {
  beforeEach(() => {
    createMockIndexedDB();
  });

  test('initializes successfully', async () => {
    const storage = new IndexedDBStorage();
    await expect(storage.initialize()).resolves.not.toThrow();
  });

  test('getItem returns null for non-existent key', async () => {
    const storage = new IndexedDBStorage();
    await storage.initialize();
    expect(storage.getItem('nonexistent')).toBeNull();
  });

  test('setItem and getItem work correctly', async () => {
    const storage = new IndexedDBStorage();
    await storage.initialize();
    storage.setItem('key1', 'value1');
    expect(storage.getItem('key1')).toBe('value1');
  });

  test('setItem overwrites existing value', async () => {
    const storage = new IndexedDBStorage();
    await storage.initialize();
    storage.setItem('key1', 'value1');
    storage.setItem('key1', 'value2');
    expect(storage.getItem('key1')).toBe('value2');
  });

  test('removeItem deletes key', async () => {
    const storage = new IndexedDBStorage();
    await storage.initialize();
    storage.setItem('key1', 'value1');
    storage.removeItem('key1');
    expect(storage.getItem('key1')).toBeNull();
  });

  test('handles uninitialized access gracefully', () => {
    const onError = jest.fn();
    const storage = new IndexedDBStorage(onError);

    expect(storage.getItem('key')).toBeNull();
    storage.setItem('key', 'value');

    expect(onError).toHaveBeenCalled();
  });

  test('handles errors with onError callback', async () => {
    const onError = jest.fn();
    const storage = new IndexedDBStorage(onError);
    await storage.initialize();

    storage.setItem('key', 'value');
    expect(onError).not.toHaveBeenCalled();
  });
});
