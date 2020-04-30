import React from 'react';
import { shallow } from 'enzyme';
import { PageNotFoundView } from './PageNotFoundView';

describe('PageNotFoundView', () => {
  test('should render without exploding', () => {
    shallow(<PageNotFoundView />);
  });
});
