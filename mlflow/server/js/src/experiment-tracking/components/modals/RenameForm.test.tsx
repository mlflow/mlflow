import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { RenameForm } from './RenameForm';
import { identity } from 'lodash';

describe('Render test', () => {
  const minimalProps = {
    type: 'run',
    name: 'Test',
    visible: true,
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn(() => identity) },
    innerRef: {},
  };

  it('should render with minimal props without exploding', () => {
    renderWithIntl(<RenameForm {...minimalProps} />);
    expect(screen.getByTestId('rename-modal-input')).toBeInTheDocument();
  });
});
