import { render } from '@testing-library/react';
import { IntlProvider } from 'react-intl';

import { TagAssignmentRow } from './TagAssignmentRow';

describe('TagAssignmentRow', () => {
  it('should throw an error if more than 3 children are passed', () => {
    const children = Array(4)
      .fill(null)
      .map((_, i) => <div key={i} />);

    const renderComponent = () =>
      render(
        <IntlProvider locale="en">
          <TagAssignmentRow>{children}</TagAssignmentRow>
        </IntlProvider>,
      );

    expect(renderComponent).toThrow('TagAssignmentRow must have 3 children or less');
  });

  it('should render children', () => {
    const children = Array(3)
      .fill(null)
      .map((_, i) => <div key={i} />);

    const { container } = render(
      <IntlProvider locale="en">
        <TagAssignmentRow>{children}</TagAssignmentRow>
      </IntlProvider>,
    );

    expect(container).toMatchSnapshot();
  });
});
