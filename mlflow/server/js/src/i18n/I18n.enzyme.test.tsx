import React from 'react';
import { FormattedMessage } from 'react-intl';
import { shallowWithIntl, mountWithIntl } from 'common/utils/TestUtils.enzyme';

function TestComponent() {
  return (
    <div data-test-id="test-component">
      <FormattedMessage
        defaultMessage="This is a default message!"
        description="Test description to ensure that the default message is rendered"
      />
    </div>
  );
}

describe('i18n', () => {
  it('mounting returns the default message without any locales generated', () => {
    const defaultMessage = 'This is a default message!';
    const wrapper = mountWithIntl(<TestComponent />);
    expect(wrapper.find('[data-test-id="test-component"]').text()).toEqual(defaultMessage);
  });

  it('shallow renders without blowing up', () => {
    const wrapper = shallowWithIntl(<TestComponent />);

    expect(wrapper.dive().find('[data-test-id="test-component"]').length).toEqual(1);
  });
});
