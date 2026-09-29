import { describe, expect, it } from '@jest/globals';
import { render, screen, within } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import type { FeedbackAssessment } from '../ModelTrace.types';
import { FeedbackItemContent } from './FeedbackItemContent';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const savedAssessment: FeedbackAssessment = {
  assessment_id: 'a-jev-1',
  assessment_name: 'quality',
  trace_id: 'tr-1',
  source: { source_type: 'LLM_JUDGE', source_id: 'typesafe:/jev-latest' },
  create_time: '2026-01-01',
  last_update_time: '2026-01-01',
  feedback: { value: false },
  metadata: { 'jev.model': 'jev-latest', 'jev.probability': '0.65' },
};

describe('FeedbackItemContent', () => {
  it('renders each saved Jev probability for assessments sharing a thresholded false decision', () => {
    render(
      <>
        <div data-testid="first-saved-assessment">
          <FeedbackItemContent feedback={savedAssessment} />
        </div>
        <div data-testid="second-saved-assessment">
          <FeedbackItemContent
            feedback={{
              ...savedAssessment,
              assessment_id: 'a-jev-2',
              metadata: { 'jev.model': 'jev-latest', 'jev.probability': '0.25' },
            }}
          />
        </div>
      </>,
      { wrapper: Wrapper },
    );

    expect(screen.getAllByText('False')).toHaveLength(2);
    expect(screen.getAllByText('Probability of true')).toHaveLength(2);
    expect(within(screen.getByTestId('first-saved-assessment')).getByText('65%')).toBeInTheDocument();
    expect(within(screen.getByTestId('second-saved-assessment')).getByText('25%')).toBeInTheDocument();
  });

  it('does not show Jev metadata for a null result', () => {
    render(<FeedbackItemContent feedback={{ ...savedAssessment, feedback: { value: null } }} />, {
      wrapper: Wrapper,
    });

    expect(screen.queryByText('Probability of true')).not.toBeInTheDocument();
  });
});
