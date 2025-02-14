import type { CSSObject } from '@emotion/react';

/**
 * Used to hide text visually, but still be readable by screen-readers
 * and other assistive devices.
 *
 * https://www.tpgi.com/the-anatomy-of-visually-hidden/
 */
export const visuallyHidden: CSSObject = {
  '&:not(:focus):not(:active)': {
    clip: 'rect(0 0 0 0)',
    clipPath: 'inset(50%)',
    height: '1px',
    overflow: 'hidden',
    position: 'absolute',
    whiteSpace: 'nowrap',
    width: '1px',
  },
};
