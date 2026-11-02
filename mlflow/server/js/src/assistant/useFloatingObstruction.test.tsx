import { describe, expect, jest, test } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useRef } from 'react';
import {
  useCoverFloatingBottomObstructions,
  useFloatingObstructionHeight,
  useRegisterFloatingBottomObstruction,
} from './useFloatingObstruction';

describe('useCoverFloatingBottomObstructions', () => {
  test('drops the bottom reservation while a surface covers the bottom-pinned bars', () => {
    const bar = document.createElement('div');
    jest.spyOn(bar, 'getBoundingClientRect').mockReturnValue({
      top: window.innerHeight - 60,
      bottom: window.innerHeight - 20,
      height: 40,
    } as DOMRect);

    const { result, rerender } = renderHook(
      ({ isCovering }: { isCovering: boolean }) => {
        const ref = useRef<HTMLElement>(bar);
        useRegisterFloatingBottomObstruction(ref);
        useCoverFloatingBottomObstructions(isCovering);
        return useFloatingObstructionHeight();
      },
      { initialProps: { isCovering: false } },
    );
    expect(result.current).toBe(60);

    rerender({ isCovering: true });
    expect(result.current).toBe(0);

    rerender({ isCovering: false });
    expect(result.current).toBe(60);
  });
});
