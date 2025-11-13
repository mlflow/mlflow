import { describe, test, expect } from '@jest/globals';
import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GetLinkModal } from './GetLinkModal';
import { DesignSystemProvider } from '@databricks/design-system';

describe('GetLinkModal', () => {
  const minimalProps: any = {
    visible: true,
    onCancel: () => {},
    link: 'link',
  };

  test('should render with minimal props without exploding', () => {
    renderWithIntl(
      <DesignSystemProvider>
        <GetLinkModal {...minimalProps} />
      </DesignSystemProvider>,
    );
    expect(screen.getByText('Get Link')).toBeInTheDocument();
  });
});
