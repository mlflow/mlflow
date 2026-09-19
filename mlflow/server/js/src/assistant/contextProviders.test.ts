import { describe, test, expect } from '@jest/globals';
import { collectDynamicAssistantContext, registerAssistantContextProvider } from './contextProviders';

describe('contextProviders', () => {
  test('collects values from registered providers', () => {
    const unregister = registerAssistantContextProvider('foo', () => ({ bar: 'baz' }));
    expect(collectDynamicAssistantContext()).toEqual({ foo: { bar: 'baz' } });
    unregister();
  });

  test('omits a key whose provider returns null or undefined', () => {
    const unregisterNull = registerAssistantContextProvider('nullKey', () => null);
    const unregisterUndefined = registerAssistantContextProvider('undefinedKey', () => undefined);
    const result = collectDynamicAssistantContext();
    expect(result).not.toHaveProperty('nullKey');
    expect(result).not.toHaveProperty('undefinedKey');
    unregisterNull();
    unregisterUndefined();
  });

  test('a later registration for the same key overwrites the earlier one', () => {
    const unregisterFirst = registerAssistantContextProvider('key', () => 'first');
    const unregisterSecond = registerAssistantContextProvider('key', () => 'second');
    expect(collectDynamicAssistantContext()).toEqual({ key: 'second' });
    unregisterSecond();
    unregisterFirst();
  });

  test('unregister is a no-op once superseded by a newer registration for the same key', () => {
    const unregisterFirst = registerAssistantContextProvider('key', () => 'first');
    const unregisterSecond = registerAssistantContextProvider('key', () => 'second');
    // Unregistering the stale (superseded) provider must not clear the current one.
    unregisterFirst();
    expect(collectDynamicAssistantContext()).toEqual({ key: 'second' });
    unregisterSecond();
  });

  test('unregister removes the key entirely', () => {
    const unregister = registerAssistantContextProvider('temp', () => 'value');
    expect(collectDynamicAssistantContext()).toEqual({ temp: 'value' });
    unregister();
    expect(collectDynamicAssistantContext()).not.toHaveProperty('temp');
  });

  test('providers are invoked lazily on every collection, not cached', () => {
    let count = 0;
    const unregister = registerAssistantContextProvider('counter', () => ++count);
    expect(collectDynamicAssistantContext()).toEqual({ counter: 1 });
    expect(collectDynamicAssistantContext()).toEqual({ counter: 2 });
    unregister();
  });
});
