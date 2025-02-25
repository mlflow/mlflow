import { act, renderHook } from '@testing-library/react';

import { useResponsiveDirection } from './Stepper';

describe('Stepper', () => {
  describe('useResponsiveDirection', () => {
    let originalRaf: typeof window.requestAnimationFrame;
    let originalResizeObserver: typeof window.ResizeObserver;
    let mockRaf: jest.Mock;
    let mockRafCallback: any;
    let mockResizeObserver: jest.Mock;

    beforeEach(() => {
      originalRaf = window.requestAnimationFrame;
      originalResizeObserver = window.ResizeObserver;

      mockRaf = jest.fn((callback) => {
        mockRafCallback = callback;
      });
      window.requestAnimationFrame = mockRaf;
    });

    afterEach(() => {
      window.requestAnimationFrame = originalRaf;
      window.ResizeObserver = originalResizeObserver;
    });

    function mockObserve(mockWidth: number) {
      mockResizeObserver = jest.fn((callback) => ({
        observe: () => callback([{ target: { clientWidth: mockWidth } }]),
        disconnect: jest.fn(),
      }));
      window.ResizeObserver = mockResizeObserver;
    }

    it('returns "vertical" for width 600 for vertical requested direction', () => {
      mockObserve(600);
      const { result } = renderHook(() =>
        useResponsiveDirection({
          requestedDirection: 'vertical',
          responsive: true,
          enabled: true,
          ref: { current: {} as any },
        }),
      );

      expect(mockRafCallback).toBeUndefined();
      expect(result.current.direction).toBe('vertical');
    });

    it('returns "horizontal" for width 600 for horizontal requested direction', () => {
      mockObserve(600);
      const { result } = renderHook(() =>
        useResponsiveDirection({
          requestedDirection: 'horizontal',
          responsive: true,
          enabled: true,
          ref: { current: {} as any },
        }),
      );

      act(() => {
        mockRafCallback();
      });

      expect(result.current.direction).toBe('horizontal');
    });

    it('returns "vertical" for width 300 for horizontal requested direction', () => {
      mockObserve(300);
      const { result } = renderHook(() =>
        useResponsiveDirection({
          requestedDirection: 'horizontal',
          responsive: true,
          enabled: true,
          ref: { current: {} as any },
        }),
      );

      act(() => {
        mockRafCallback();
      });

      expect(result.current.direction).toBe('vertical');
    });
  });
});
