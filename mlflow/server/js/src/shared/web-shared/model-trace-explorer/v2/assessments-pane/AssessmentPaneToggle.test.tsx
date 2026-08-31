import { describe, expect, it } from '@jest/globals';
import { screen } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render } from '@databricks/web-shared/test-utils/render';

import { AssessmentPaneToggle } from './AssessmentPaneToggle';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('AssessmentPaneToggle', () => {
  it('shows the assessment label and count in wide mode', () => {
    render(<AssessmentPaneToggle assessmentCount={3} />, { wrapper: Wrapper });

    expect(screen.getByRole('button', { name: 'Assess trace (3 assessments)' })).toBeInTheDocument();
    expect(screen.getByText('Assess')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('keeps the assessment count visible in compact mode', () => {
    render(<AssessmentPaneToggle assessmentCount={3} compact />, { wrapper: Wrapper });

    expect(screen.getByRole('button', { name: 'Assess trace (3 assessments)' })).toBeInTheDocument();
    expect(screen.queryByText('Assess')).not.toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });
});
