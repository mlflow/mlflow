import { describe, it, expect } from '@jest/globals';
import React from 'react';
import { CopyBox } from './CopyBox';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

describe('CopyBox', () => {
  it('should render with minimal props without exploding', () => {
    renderWithIntl(
      <DesignSystemProvider>
        <CopyBox copyText="copy text" />
      </DesignSystemProvider>,
    );
    const input = screen.getByTestId('copy-box');
    expect(input).toHaveValue('copy text');
    expect(input).toHaveAttribute('readOnly');
  });
});
