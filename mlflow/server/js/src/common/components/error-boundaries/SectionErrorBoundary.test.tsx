/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { SectionErrorBoundary } from './SectionErrorBoundary';
import { SupportPageUrl } from '../../constants';

describe('SectionErrorBoundary', () => {
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = { children: 'testChild' };
  });

  test('should render with minimal props without exploding', () => {
    renderWithIntl(<SectionErrorBoundary {...minimalProps} />);
    expect(screen.getByText('testChild')).toBeInTheDocument();
    expect(screen.queryByTestId('icon-fail')).not.toBeInTheDocument();
  });

  test('componentDidCatch causes error message to render', () => {
    const ErrorComponent = () => {
      throw new Error('error msg');
    };
    renderWithIntl(
      <SectionErrorBoundary {...minimalProps}>
        <ErrorComponent />{' '}
      </SectionErrorBoundary>,
    );

    expect(screen.getByTestId('icon-fail')).toBeInTheDocument();
    expect(screen.queryByText('testChild')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /here/i })).toHaveAttribute('href', SupportPageUrl);
  });

  test('should show error if showServerError prop passed in', () => {
    const ErrorComponent = () => {
      throw new Error('some error message');
    };
    renderWithIntl(
      <SectionErrorBoundary {...minimalProps} showServerError>
        <ErrorComponent />{' '}
      </SectionErrorBoundary>,
    );

    expect(screen.getByText(/error message: some error message/i)).toBeInTheDocument();
  });
});
