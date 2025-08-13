import React from 'react';
import { FormattedMessage } from './i18n';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

function TestComponent() {
  return (
    <div data-testid="test-component">
      <FormattedMessage
        defaultMessage="This is a default message!"
        description="Test description to ensure that the default message is rendered"
      />
    </div>
  );
}

describe('i18n', () => {
  it('render returns the default message without any locales generated', () => {
    const defaultMessage = 'This is a default message!';
    renderWithIntl(<TestComponent />);
    expect(screen.getByTestId('test-component')).toHaveTextContent(defaultMessage);
  });
});
