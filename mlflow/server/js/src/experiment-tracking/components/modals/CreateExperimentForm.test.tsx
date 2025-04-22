import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { CreateExperimentForm } from './CreateExperimentForm';

describe('Render test', () => {
  const minimalProps = {
    visible: true,
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c: any) => c) },
  };

  it('should render with minimal props without exploding', () => {
    renderWithIntl(<CreateExperimentForm {...minimalProps} />);
    expect(
      screen.getByRole('textbox', {
        name: /experiment name/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('textbox', {
        name: /artifact location/i,
      }),
    ).toBeInTheDocument();
  });
});
