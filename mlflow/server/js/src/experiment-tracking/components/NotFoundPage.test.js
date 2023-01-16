import React from 'react';
import { shallow } from 'enzyme';
import NotFoundPage from './NotFoundPage';

describe('NotFoundPage', () => {
  test('should render without exploding', () => {
    shallow(<NotFoundPage />);
  });
});
