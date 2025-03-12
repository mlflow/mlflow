import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ErrorModalWithIntl } from './ErrorModal';

describe('ErrorModalImpl', () => {
  const minimalProps: any = {
    isOpen: true,
    onClose: jest.fn(),
    text: 'Error popup content',
  };

  test('should render with minimal props without exploding', () => {
    renderWithIntl(<ErrorModalWithIntl {...minimalProps} />);
    expect(screen.getByText(/error popup content/i)).toBeInTheDocument();
  });
});
