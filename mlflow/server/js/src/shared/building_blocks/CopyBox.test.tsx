import React from 'react';
import { CopyBox } from './CopyBox';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

describe('CopyBox', () => {
  it('should render with minimal props without exploding', () => {
    renderWithIntl(<CopyBox copyText="copy text" />);
    const input = screen.getByTestId('copy-box');
    expect(input).toHaveValue('copy text');
    expect(input).toHaveAttribute('readOnly');
  });
});
