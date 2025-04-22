import React from 'react';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import NotFoundPage from './NotFoundPage';

describe('NotFoundPage', () => {
  test('should render without exploding', () => {
    render(<NotFoundPage />);
    expect(screen.getByText('Resource not found.')).toBeInTheDocument();
  });
});
