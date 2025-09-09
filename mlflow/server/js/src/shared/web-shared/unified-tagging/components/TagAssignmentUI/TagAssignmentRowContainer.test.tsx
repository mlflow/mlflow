import { render, screen } from '@testing-library/react';

import { TagAssignmentRowContainer } from './TagAssignmentRowContainer';

describe('TagAssignmentRowContainer', () => {
  it('should render children', () => {
    render(
      <TagAssignmentRowContainer>
        <div>child</div>
      </TagAssignmentRowContainer>,
    );

    expect(screen.getByText('child')).toBeInTheDocument();
  });
});
