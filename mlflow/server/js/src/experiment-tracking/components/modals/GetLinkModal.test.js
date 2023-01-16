import React from 'react';
import { shallow } from 'enzyme';
import { GetLinkModal } from './GetLinkModal';

describe('GetLinkModal', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      visible: true,
      onCancel: () => {},
      link: 'link',
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<GetLinkModal {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});
