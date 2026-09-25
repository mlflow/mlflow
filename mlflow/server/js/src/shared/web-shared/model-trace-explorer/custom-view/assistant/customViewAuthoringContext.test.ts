import { describe, test, expect } from '@jest/globals';
import {
  registerCustomViewAuthoringContext,
  getCustomViewAuthoringContext,
  latchDispatchedCustomViewApplyTarget,
  getDispatchedCustomViewApplyTarget,
  type CustomViewAuthoringContext,
} from './customViewAuthoringContext';

const makeContext = (overrides: Partial<CustomViewAuthoringContext> = {}): CustomViewAuthoringContext => ({
  traceSample: { metrics: { status: 'OK' } },
  ...overrides,
});

describe('customViewAuthoringContext', () => {
  test('returns null when nothing is registered', () => {
    expect(getCustomViewAuthoringContext()).toBeNull();
  });

  test('registers and reads back the context', () => {
    const context = makeContext();
    const unregister = registerCustomViewAuthoringContext(context);
    expect(getCustomViewAuthoringContext()).toBe(context);
    unregister();
  });

  test('unregister clears the context', () => {
    const unregister = registerCustomViewAuthoringContext(makeContext());
    unregister();
    expect(getCustomViewAuthoringContext()).toBeNull();
  });

  test('a later registration overwrites the earlier one (last-writer-wins)', () => {
    const first = makeContext({ traceSample: { id: 'first' } });
    const second = makeContext({ traceSample: { id: 'second' } });
    const unregisterFirst = registerCustomViewAuthoringContext(first);
    const unregisterSecond = registerCustomViewAuthoringContext(second);

    expect(getCustomViewAuthoringContext()).toBe(second);

    unregisterSecond();
    unregisterFirst();
  });

  test('unregistering a superseded registration is a no-op (does not clear the current one)', () => {
    const first = makeContext({ traceSample: { id: 'first' } });
    const second = makeContext({ traceSample: { id: 'second' } });
    const unregisterFirst = registerCustomViewAuthoringContext(first);
    const unregisterSecond = registerCustomViewAuthoringContext(second);

    unregisterFirst();
    expect(getCustomViewAuthoringContext()).toBe(second);

    unregisterSecond();
    expect(getCustomViewAuthoringContext()).toBeNull();
  });
});

describe('latchDispatchedCustomViewApplyTarget / getDispatchedCustomViewApplyTarget', () => {
  test('returns undefined before any turn has latched a target', () => {
    latchDispatchedCustomViewApplyTarget(undefined);
    expect(getDispatchedCustomViewApplyTarget()).toBeUndefined();
  });

  test('latches the target passed on the most recent prompt-assembly call', () => {
    const target = { id: 'view-1', name: 'My view', instruction: 'show text', createdAtMs: 1 };
    latchDispatchedCustomViewApplyTarget(target);
    expect(getDispatchedCustomViewApplyTarget()).toBe(target);
  });

  test('a later latch overwrites the earlier one, including clearing it with undefined', () => {
    const first = { id: 'view-1', name: 'First', instruction: '', createdAtMs: 1 };
    latchDispatchedCustomViewApplyTarget(first);
    expect(getDispatchedCustomViewApplyTarget()).toBe(first);

    const second = { id: 'view-2', name: 'Second', instruction: '', createdAtMs: 2 };
    latchDispatchedCustomViewApplyTarget(second);
    expect(getDispatchedCustomViewApplyTarget()).toBe(second);

    latchDispatchedCustomViewApplyTarget(undefined);
    expect(getDispatchedCustomViewApplyTarget()).toBeUndefined();
  });
});
