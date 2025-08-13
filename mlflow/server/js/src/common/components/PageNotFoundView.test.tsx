import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { BrowserRouter } from '../utils/RoutingUtils';
import { PageNotFoundView } from './PageNotFoundView';

describe('PageNotFoundView', () => {
  test('should render without exploding', () => {
    renderWithIntl(
      <BrowserRouter>
        <PageNotFoundView />
      </BrowserRouter>,
    );
    expect(screen.getByText('Page Not Found')).toBeInTheDocument();
  });
});
