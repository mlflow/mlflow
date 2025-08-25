import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { TagAssignmentRemoveButtonUI } from './TagAssignmentRemoveButtonUI';

describe('TagAssignmentRemoveButtonUI', () => {
  it('should render a button', async () => {
    const handleClick = jest.fn();
    render(<TagAssignmentRemoveButtonUI componentId="test" onClick={handleClick} />);

    const button = screen.getByRole('button');
    await userEvent.click(button);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
