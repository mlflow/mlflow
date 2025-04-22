import React from 'react';
import { CreateModelForm } from './CreateModelForm';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

describe('Render test', () => {
  const minimalProps = {
    visible: true,
    form: { getFieldDecorator: jest.fn(() => (c: any) => c) },
  };

  test('should render form in modal', () => {
    renderWithIntl(<CreateModelForm {...minimalProps} />);
    expect(screen.getByTestId('create-model-form-modal')).toBeInTheDocument();
  });
});
