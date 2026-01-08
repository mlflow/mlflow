import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { SingleChatTurnAssessments } from './SingleChatTurnAssessments';
import type { Assessment, ModelTrace } from '../ModelTrace.types';
import { ASSESSMENT_SESSION_METADATA_KEY } from '../constants';

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('SingleChatTurnAssessments', () => {
  const mockGetAssessmentTitle = jest.fn((name: string) => name);
  const mockOnAddAssessmentsClick = jest.fn();

  const createMockTrace = (assessments: Assessment[]): ModelTrace => ({
    data: { spans: [] },
    info: {
      trace_id: 'trace-1',
      request_time: '2025-04-19T09:04:07.875Z',
      state: 'OK',
      tags: {},
      assessments,
      trace_location: {
        type: 'MLFLOW_EXPERIMENT',
        mlflow_experiment: {
          experiment_id: 'exp-1',
        },
      },
    },
  });

  const createMockAssessment = (id: string, name: string, overrides?: Partial<Assessment>): Assessment => ({
    assessment_id: id,
    assessment_name: name,
    trace_id: 'trace-1',
    span_id: '',
    source: {
      source_type: 'HUMAN',
      source_id: 'test@databricks.com',
    },
    create_time: '2025-04-19T09:04:07.875Z',
    last_update_time: '2025-04-19T09:04:07.875Z',
    feedback: {
      value: 'yes',
    },
    ...overrides,
  });

  const createSessionLevelAssessment = (id: string, name: string): Assessment =>
    createMockAssessment(id, name, {
      metadata: {
        [ASSESSMENT_SESSION_METADATA_KEY]: 'session-123',
      },
    });

  it('should show empty state when no trace-level assessments exist', () => {
    const trace = createMockTrace([]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    expect(screen.getByText('Evaluate trace')).toBeInTheDocument();
  });

  it('should display trace-level assessments', () => {
    const trace = createMockTrace([
      createMockAssessment('trace-1', 'Relevance'),
      createMockAssessment('trace-2', 'Correctness'),
    ]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    expect(screen.getByText('Relevance:', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Correctness:', { exact: false })).toBeInTheDocument();
    expect(screen.queryByText('Evaluate trace')).not.toBeInTheDocument();
  });

  it('should filter out session-level assessments from displayed assessments', () => {
    const trace = createMockTrace([
      createMockAssessment('trace-1', 'Relevance'),
      createSessionLevelAssessment('session-1', 'SessionRelevance'),
      createMockAssessment('trace-2', 'Correctness'),
    ]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    expect(screen.getByText('Relevance:', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Correctness:', { exact: false })).toBeInTheDocument();
    expect(screen.queryByText('SessionRelevance:', { exact: false })).not.toBeInTheDocument();
  });

  it('should filter out invalid assessments from displayed assessments', () => {
    const trace = createMockTrace([
      createMockAssessment('trace-1', 'Relevance'),
      createMockAssessment('invalid-1', 'InvalidAssessment', { valid: false }),
      createMockAssessment('trace-2', 'Correctness'),
    ]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    expect(screen.getByText('Relevance:', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Correctness:', { exact: false })).toBeInTheDocument();
    expect(screen.queryByText('InvalidAssessment:', { exact: false })).not.toBeInTheDocument();
  });

  it('should group assessments by name and show only the most recent', () => {
    const trace = createMockTrace([
      createMockAssessment('old-1', 'Relevance', {
        create_time: '2025-04-19T08:00:00.000Z',
      }),
      createMockAssessment('new-1', 'Relevance', {
        create_time: '2025-04-19T10:00:00.000Z',
      }),
      createMockAssessment('middle-1', 'Relevance', {
        create_time: '2025-04-19T09:00:00.000Z',
      }),
    ]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    // Should only render one instance of "Relevance" (the most recent one)
    const relevanceElements = screen.getAllByText('Relevance:', { exact: false });
    expect(relevanceElements).toHaveLength(1);
  });

  it('should limit displayed assessments to 5', () => {
    const trace = createMockTrace([
      createMockAssessment('1', 'Assessment1'),
      createMockAssessment('2', 'Assessment2'),
      createMockAssessment('3', 'Assessment3'),
      createMockAssessment('4', 'Assessment4'),
      createMockAssessment('5', 'Assessment5'),
      createMockAssessment('6', 'Assessment6'),
      createMockAssessment('7', 'Assessment7'),
    ]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    // Should only show first 5 assessments
    expect(screen.getByText('Assessment1:', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Assessment2:', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Assessment3:', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Assessment4:', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Assessment5:', { exact: false })).toBeInTheDocument();
    expect(screen.queryByText('Assessment6:', { exact: false })).not.toBeInTheDocument();
    expect(screen.queryByText('Assessment7:', { exact: false })).not.toBeInTheDocument();
  });

  it('should show indicator when more than 5 assessments exist', () => {
    const trace = createMockTrace([
      createMockAssessment('1', 'Assessment1'),
      createMockAssessment('2', 'Assessment2'),
      createMockAssessment('3', 'Assessment3'),
      createMockAssessment('4', 'Assessment4'),
      createMockAssessment('5', 'Assessment5'),
      createMockAssessment('6', 'Assessment6'),
      createMockAssessment('7', 'Assessment7'),
      createMockAssessment('8', 'Assessment8'),
    ]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    // Should show "+3" indicator for 3 remaining assessments (8 total - 5 shown)
    // The Overflow component uses "+N" format (no space)
    expect(screen.getByText('+3')).toBeInTheDocument();
  });

  it('should not show indicator when 5 or fewer assessments exist', () => {
    const trace = createMockTrace([
      createMockAssessment('1', 'Assessment1'),
      createMockAssessment('2', 'Assessment2'),
      createMockAssessment('3', 'Assessment3'),
      createMockAssessment('4', 'Assessment4'),
      createMockAssessment('5', 'Assessment5'),
    ]);

    render(
      <TestWrapper>
        <SingleChatTurnAssessments
          trace={trace}
          getAssessmentTitle={mockGetAssessmentTitle}
          onAddAssessmentsClick={mockOnAddAssessmentsClick}
        />
      </TestWrapper>,
    );

    // Should not show any "+ N" indicator
    expect(screen.queryByText(/\+ \d+/)).not.toBeInTheDocument();
  });
});
