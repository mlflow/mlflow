import React from 'react';
import { shallow } from 'enzyme';
import PreviewIcon from './PreviewIcon';

describe('PreviewIcon', () => {
  let wrapper;

  beforeEach(() => {
    wrapper = shallow(<PreviewIcon className='test-class' />);
  });

  describe('Rendering', () => {
    it('should render PreviewIcon with className', () => {
      expect(wrapper.find('span').prop('className')).toContain('test-class');
    });
  });
});
