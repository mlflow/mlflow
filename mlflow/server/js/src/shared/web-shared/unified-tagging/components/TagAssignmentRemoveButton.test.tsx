import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';

import { TagAssignmentRemoveButton } from './TagAssignmentRemoveButton';
import { TestTagAssignmentContextProvider } from '../test-utils/TestTagAssignmentContextProvider';

describe('TagAssignmentRemoveButton', () => {
  it('should call function with right index when clicked', async () => {
    const removeOrUpdate = jest.fn();
    render(
      <IntlProvider locale="en">
        <TestTagAssignmentContextProvider removeOrUpdate={removeOrUpdate}>
          <TagAssignmentRemoveButton index={1} componentId="test" />
        </TestTagAssignmentContextProvider>
      </IntlProvider>,
    );

    expect(removeOrUpdate).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button'));

    expect(removeOrUpdate).toHaveBeenCalledWith(1);
  });
});
