import React from 'react';
import { CopyBox } from './CopyBox';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';

describe('CopyBox', () => {
  it('should render with minimal props without exploding', () => {
    const wrapper = mountWithIntl(<CopyBox copyText="copy text" />);
    const input = wrapper.find("input[data-test-id='copy-box']");
    expect(input.props().value).toEqual('copy text');
    expect(input.props().readOnly).toBe(true);
  });
});
