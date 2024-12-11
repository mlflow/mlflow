import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { Spinner } from './Spinner';

describe('Spinner', () => {
  test('should render with no props without exploding', () => {
    renderWithIntl(<Spinner />);
    expect(screen.getByAltText('Page loading...')).toBeInTheDocument();
  });
});
