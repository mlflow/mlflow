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

  test('sees a change made before an await', async () => {
    const style = document.createElement('style');
    style.textContent = '.collapsed { visibility: hidden; }';
    document.head.appendChild(style);
    const element = document.createElement('section');
    document.body.appendChild(element);
    expect(window.getComputedStyle(element).visibility).toBe('visible');

    element.className = 'collapsed';
    await Promise.resolve();
    expect(window.getComputedStyle(element).visibility).toBe('hidden');
  });

  test('sees focus and checked state that a rule depends on', () => {
    const style = document.createElement('style');
    style.textContent = 'button:focus { visibility: hidden; } input:checked { display: none; }';
    document.head.appendChild(style);
    const button = document.createElement('button');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    document.body.append(button, checkbox);
    expect(window.getComputedStyle(button).visibility).toBe('visible');
    expect(window.getComputedStyle(checkbox).display).toBe('inline-block');

    button.focus();
    checkbox.checked = true;
    expect(window.getComputedStyle(button).visibility).toBe('hidden');
    expect(window.getComputedStyle(checkbox).display).toBe('none');
  });

  test('sees a rule inserted into an existing stylesheet', () => {
    const style = document.createElement('style');
    document.head.appendChild(style);
    const element = document.createElement('p');
    element.className = 'inserted';
    document.body.appendChild(element);
    expect(window.getComputedStyle(element).display).toBe('block');

    style.sheet?.insertRule('.inserted { display: none; }');
    expect(window.getComputedStyle(element).display).toBe('none');
  });
});
