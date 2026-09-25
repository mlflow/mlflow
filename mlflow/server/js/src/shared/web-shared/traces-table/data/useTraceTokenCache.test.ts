import { describe, expect, test } from '@jest/globals';
import { PageTokenStack } from './useTraceTokenCache';

describe('PageTokenStack', () => {
  test('page 1 needs no token and is always known', () => {
    const stack = new PageTokenStack();
    expect(stack.getTokenForPage(1)).toBeUndefined();
    expect(stack.isPageKnown(1)).toBe(true);
    expect(stack.hasPrev(1)).toBe(false);
  });

  test('records a next token so the following page becomes reachable', () => {
    const stack = new PageTokenStack();
    expect(stack.isPageKnown(2)).toBe(false);
    expect(stack.getTokenForPage(2)).toBeUndefined();

    stack.recordNextToken(1, 'token-for-page-2');
    expect(stack.isPageKnown(2)).toBe(true);
    expect(stack.getTokenForPage(2)).toBe('token-for-page-2');
    expect(stack.hasNext(1)).toBe(true);
  });

  test('a null next token marks the last page', () => {
    const stack = new PageTokenStack();
    stack.recordNextToken(1, 'token-for-page-2');
    stack.recordNextToken(2, undefined); // page 2 is the last page
    expect(stack.hasNext(2)).toBe(false);
    expect(stack.isPageKnown(3)).toBe(false);
    expect(stack.getTokenForPage(3)).toBeUndefined();
  });

  test('an empty-string next token is treated as the last page (proto3 default, not a real cursor)', () => {
    const stack = new PageTokenStack();
    stack.recordNextToken(1, '');
    expect(stack.hasNext(1)).toBe(false);
    expect(stack.isPageKnown(2)).toBe(false);
    expect(stack.getTokenForPage(2)).toBeUndefined();
  });

  test('hasNext is optimistic for a not-yet-resolved page', () => {
    const stack = new PageTokenStack();
    stack.recordNextToken(1, 'token-for-page-2');
    expect(stack.hasNext(2)).toBe(true);
  });

  test('recorded tokens chain sequentially across pages', () => {
    const stack = new PageTokenStack();
    stack.recordNextToken(1, 't2');
    stack.recordNextToken(2, 't3');
    stack.recordNextToken(3, 't4');
    expect(stack.getTokenForPage(2)).toBe('t2');
    expect(stack.getTokenForPage(3)).toBe('t3');
    expect(stack.getTokenForPage(4)).toBe('t4');
  });

  test('resetIfKeyChanged clears the stack only when the key changes', () => {
    const stack = new PageTokenStack();
    stack.resetIfKeyChanged('key-a');
    stack.recordNextToken(1, 't2');
    expect(stack.getTokenForPage(2)).toBe('t2');

    stack.resetIfKeyChanged('key-a');
    expect(stack.getTokenForPage(2)).toBe('t2');

    stack.resetIfKeyChanged('key-b');
    expect(stack.getTokenForPage(2)).toBeUndefined();
    expect(stack.isPageKnown(2)).toBe(false);
    expect(stack.getKey()).toBe('key-b');
  });

  test('hasPrev is purely positional', () => {
    const stack = new PageTokenStack();
    expect(stack.hasPrev(1)).toBe(false);
    expect(stack.hasPrev(2)).toBe(true);
    expect(stack.hasPrev(5)).toBe(true);
  });
});
