import { describe, it, expect, afterEach, jest } from '@jest/globals';
import { copyToClipboard } from './copyToClipboard';

describe('copyToClipboard', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses navigator.clipboard.writeText in secure contexts', async () => {
    const writeText = jest.fn().mockImplementation(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    const result = await copyToClipboard('hello');

    expect(writeText).toHaveBeenCalledWith('hello');
    expect(result).toBe(true);
  });

  it('falls back to execCommand when clipboard API is unavailable (HTTP context)', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: undefined,
      configurable: true,
    });
    const execCommand = jest.fn().mockReturnValue(true);
    Object.defineProperty(document, 'execCommand', { value: execCommand, configurable: true });

    const result = await copyToClipboard('hello');

    expect(execCommand).toHaveBeenCalledWith('copy');
    expect(result).toBe(true);
  });

  it('returns false if both methods fail', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: undefined,
      configurable: true,
    });
    Object.defineProperty(document, 'execCommand', {
      value: jest.fn().mockImplementation(() => {
        throw new Error('not supported');
      }),
      configurable: true,
    });

    const result = await copyToClipboard('hello');

    expect(result).toBe(false);
  });
});
