import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GetLinkModal } from './GetLinkModal';

describe('GetLinkModal', () => {
  const minimalProps: any = {
    visible: true,
    onCancel: () => {},
    link: 'link',
  };

  test('should render with minimal props without exploding', () => {
    renderWithIntl(<GetLinkModal {...minimalProps} />);
    expect(screen.getByText('Get Link')).toBeInTheDocument();
  });
});
