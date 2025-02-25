import { render, screen } from '@testing-library/react';

import { describe, expect, it } from '@databricks/config-jest';

import { TableSkeleton } from './TableSkeleton';

describe('TableSkeleton', () => {
  it('should indicate loading status via aria-busy', () => {
    render(<TableSkeleton />);
    expect(screen.getByRole('status')).toHaveAttribute('aria-busy', 'true');
  });
});
