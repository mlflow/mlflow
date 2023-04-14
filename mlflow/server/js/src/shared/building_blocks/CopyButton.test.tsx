import React from 'react';
import { CopyButton } from './CopyButton';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('CopyButton', () => {
  const originalClipboard = { ...global.navigator.clipboard };
  beforeEach(() => {
    const mockClipboard = {
      writeText: jest.fn(),
    };
    // @ts-expect-error TS(2540): Cannot assign to 'clipboard' because it is a read-... Remove this comment to see the full error message
    global.navigator.clipboard = mockClipboard;
  });

  afterEach(() => {
    jest.resetAllMocks();
    // @ts-expect-error TS(2540): Cannot assign to 'clipboard' because it is a read-... Remove this comment to see the full error message
    global.navigator.clipboard = originalClipboard;
  });

  it('should render with minimal props without exploding', () => {
    const wrapper = mountWithIntl(<CopyButton copyText='copyText' />);
    expect(wrapper.text().includes('Copy')).toBe(true);
    wrapper.simulate('click');
    expect(wrapper.text().includes('Copied')).toBe(true);
    expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('copyText');
  });
});
