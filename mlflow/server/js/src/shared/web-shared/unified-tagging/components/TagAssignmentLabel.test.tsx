import { render } from '@testing-library/react';

import { TagAssignmentLabel } from './TagAssignmentLabel';

describe('TagAssignmentLabel', () => {
  it('renders children', () => {
    const { getByText } = render(<TagAssignmentLabel>test</TagAssignmentLabel>);
    expect(getByText('test')).toBeInTheDocument();
  });
});
