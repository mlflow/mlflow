import { describe, expect, test } from '@jest/globals';
import { cacheComputedStylesUntilDomChanges } from './cacheComputedStyles';

cacheComputedStylesUntilDomChanges();

describe('cacheComputedStylesUntilDomChanges', () => {
  test('reuses an element’s computed style until the document changes', () => {
    const style = document.createElement('style');
    style.textContent = '.hidden { display: none; }';
    document.head.appendChild(style);
    const element = document.createElement('div');
    document.body.appendChild(element);

    const first = window.getComputedStyle(element);
    expect(window.getComputedStyle(element)).toBe(first);
    expect(first.display).toBe('block');

    element.className = 'hidden';
    expect(window.getComputedStyle(element).display).toBe('none');
    expect(element).not.toBeVisible();
  });

  test('sees a stylesheet added after a cached read', () => {
    const element = document.createElement('span');
    element.className = 'late';
    document.body.appendChild(element);
    expect(window.getComputedStyle(element).visibility).toBe('visible');

    const style = document.createElement('style');
    style.textContent = '.late { visibility: hidden; }';
    document.head.appendChild(style);
    expect(window.getComputedStyle(element).visibility).toBe('hidden');
  });
});
