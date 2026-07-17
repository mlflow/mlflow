import { describe, test, expect, jest } from '@jest/globals';
import { toRGBA } from './toRGBA'; // Adjust this import path as needed

describe('toRGBA', () => {
  // Test RGB input
  test('converts rgb color to rgba', () => {
    expect(toRGBA('rgb(255, 0, 0)', 0.5)).toBe('rgba(255, 0, 0, 0.5)');
  });

  test('converts rgba color to rgba with new alpha', () => {
    expect(toRGBA('rgba(0, 255, 0, 0.8)', 0.3)).toBe('rgba(0, 255, 0, 0.3)');
  });

  // Test Hex input
  test('converts 6-digit hex color to rgba', () => {
    expect(toRGBA('#00ff00', 0.7)).toBe('rgba(0, 255, 0, 0.7)');
  });

  test('converts 3-digit hex color to rgba', () => {
    expect(toRGBA('#f00', 0.2)).toBe('rgba(255, 0, 0, 0.2)');
  });

  // Test named colors
  test('converts named color to rgba', () => {
    // Mock the canvas and context
    const mockContext = {
      fillStyle: '',
    };
    const mockCanvas = {
      getContext: jest.fn(() => mockContext),
    };
    jest.spyOn(document, 'createElement').mockReturnValue(mockCanvas as any);

    // Simulate how the canvas context behaves with a named color. A little silly since we're just testing the functionality we've mocked, but named colors are very unlikely to come from the DSL anyways
    Object.defineProperty(mockContext, 'fillStyle', {
      set(value) {
        if (value === 'blue') {
          this._fillStyle = 'rgb(0, 0, 255)';
        } else {
          this._fillStyle = value;
        }
      },
      get() {
        return this._fillStyle;
      },
    });

    expect(toRGBA('blue', 0.9)).toBe('rgba(0, 0, 255, 0.9)');

    // Verify canvas methods were called
    expect(document.createElement).toHaveBeenCalledWith('canvas');
    // @ts-expect-error Expected 0 arguments, but got 1
    expect(mockCanvas.getContext).toHaveBeenCalledWith('2d');
  });

  // Test edge cases
  test('handles invalid rgb input', () => {
    expect(toRGBA('rgb(255)', 0.5)).toBe('rgba(255, 0, 0, 0.5)');
  });

  test('handles invalid hex input', () => {
    expect(toRGBA('#gg0000', 0.5)).toBe('rgba(0, 0, 0, 0.5)');
  });

  test('handles alpha value of 0', () => {
    expect(toRGBA('#ff0000', 0)).toBe('rgba(255, 0, 0, 0)');
  });

  test('handles alpha value of 1', () => {
    expect(toRGBA('#ff0000', 1)).toBe('rgba(255, 0, 0, 1)');
  });

  // New test for handling failure to get context
  test('handles failure to get canvas context', () => {
    const mockCanvas = {
      getContext: jest.fn(() => null),
    };
    jest.spyOn(document, 'createElement').mockReturnValue(mockCanvas as any);

    expect(toRGBA('somecolor', 0.5)).toBe('rgba(0, 0, 0, 0.5)');
  });
});
