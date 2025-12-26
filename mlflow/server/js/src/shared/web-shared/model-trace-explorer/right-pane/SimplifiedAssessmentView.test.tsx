import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { SimplifiedAssessmentView } from './SimplifiedAssessmentView';
import type { Assessment } from '../ModelTrace.types';
import {
  MOCK_ASSESSMENT,
  MOCK_EXPECTATION,
  MOCK_ROOT_ASSESSMENT,
  MOCK_SPAN_ASSESSMENT,
} from '../ModelTraceExplorer.test-utils';

const MOCK_ASSESSMENT_WITH_ERROR: Assessment = {
  assessment_id: 'a-test-error',
  assessment_name: 'Failed Assessment',
  trace_id: 'tr-test-v3',
  span_id: '',
  source: {
    source_type: 'LLM_JUDGE',
    source_id: '1',
  },
  create_time: '2025-04-19T09:04:07.875Z',
  last_update_time: '2025-04-19T09:04:07.875Z',
  feedback: {
    error: {
      error_code: 'EVALUATION_ERROR',
      error_message: 'Failed to evaluate assessment',
      stack_trace: 'Error stack trace here',
    },
  },
};

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('SimplifiedAssessmentView', () => {
  it('renders "No assessments available" message when assessments array is empty', () => {
    render(<SimplifiedAssessmentView assessments={[]} />, { wrapper: Wrapper });

    expect(screen.getByText('No assessments available')).toBeInTheDocument();
  });

  it('filters to show only valid feedback assessments', () => {
    const assessments: Assessment[] = [
      MOCK_ASSESSMENT, // valid feedback assessment
      MOCK_EXPECTATION, // expectation assessment - should be filtered out
      MOCK_SPAN_ASSESSMENT, // valid feedback assessment
    ];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // Should show feedback assessments
    expect(screen.getByText('Relevance')).toBeInTheDocument();
    expect(screen.getByText('Thumbs')).toBeInTheDocument();

    // Should NOT show expectation assessment
    expect(screen.queryByText('expected_facts')).not.toBeInTheDocument();
  });

  it('filters out invalid assessments (valid: false)', () => {
    const assessments: Assessment[] = [
      MOCK_ASSESSMENT, // valid assessment
      MOCK_ROOT_ASSESSMENT, // invalid assessment (valid: false)
    ];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // Should show only the valid assessment
    expect(screen.getByText('Relevance')).toBeInTheDocument();

    // Should have only one assessment card
    const assessmentCards = screen.queryAllByText(/Relevance|Thumbs/);
    expect(assessmentCards).toHaveLength(1);
  });

  it('renders an AssessmentCard for each valid feedback assessment', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT, MOCK_SPAN_ASSESSMENT];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // Should render both assessment names
    expect(screen.getByText('Relevance')).toBeInTheDocument();
    expect(screen.getByText('Thumbs')).toBeInTheDocument();
  });

  it('displays assessment value', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // The value "5" should be displayed
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('displays rationale when present', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // Check that rationale is rendered
    expect(screen.getByText('The thought process is sound and follows from the request')).toBeInTheDocument();
  });

  it('displays error using FeedbackErrorItem when feedback has error', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT_WITH_ERROR];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // Check that assessment name is shown
    expect(screen.getByText('Failed Assessment')).toBeInTheDocument();

    // Check that error message is displayed
    expect(screen.getByText('Failed to evaluate assessment')).toBeInTheDocument();
  });

  it('does not show value when error is present', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT_WITH_ERROR];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // Error message should be shown
    expect(screen.getByText('Failed to evaluate assessment')).toBeInTheDocument();

    // But no value should be displayed (there's no value in the mock anyway)
    expect(screen.queryByText('5')).not.toBeInTheDocument();
  });

  it('renders correct number of assessment cards', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT, MOCK_SPAN_ASSESSMENT, MOCK_ROOT_ASSESSMENT];

    render(<SimplifiedAssessmentView assessments={assessments} />, { wrapper: Wrapper });

    // Should render 2 cards (MOCK_ASSESSMENT and MOCK_SPAN_ASSESSMENT)
    // MOCK_ROOT_ASSESSMENT should be filtered out because valid: false
    expect(screen.getByText('Relevance')).toBeInTheDocument();
    expect(screen.getByText('Thumbs')).toBeInTheDocument();

    // Should have exactly 2 assessment names visible
    const relevanceElements = screen.getAllByText('Relevance');
    const thumbsElements = screen.getAllByText('Thumbs');
    expect(relevanceElements).toHaveLength(1);
    expect(thumbsElements).toHaveLength(1);
  });

  it('handles assessment without rationale', () => {
    const assessmentWithoutRationale: Assessment = {
      ...MOCK_ASSESSMENT,
      rationale: undefined,
    };

    render(<SimplifiedAssessmentView assessments={[assessmentWithoutRationale]} />, { wrapper: Wrapper });

    // Assessment should still render
    expect(screen.getByText('Relevance')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();

    // But rationale should not be present
    expect(screen.queryByText('The thought process is sound and follows from the request')).not.toBeInTheDocument();
  });
});
