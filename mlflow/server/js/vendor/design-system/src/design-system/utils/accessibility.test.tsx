import { render, screen } from '@testing-library/react';

import { visuallyHidden } from './accessibility';

describe('accessibility utilities', () => {
  describe('visuallyHidden', () => {
    it('should still be exposed in the accName and description', () => {
      render(
        <>
          <p css={visuallyHidden} id="desc">
            description
          </p>
          <button aria-describedby="desc">
            <span css={visuallyHidden}>label</span>
          </button>
        </>,
      );
      const button = screen.getByRole('button');
      expect(button).toHaveAccessibleName('label');
      expect(button).toHaveAccessibleDescription('description');
    });
  });
});
