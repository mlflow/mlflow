import { describe, expect, it, jest, beforeEach } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import type { RunEvaluationTracesDataEntry } from '@databricks/web-shared/genai-traces-table';
import { RunViewEvaluationsTabArtifacts } from './RunViewEvaluationsTabArtifacts';

const mockSetCompareToRunUuid = jest.fn();
let mockShouldEnableImprovedEvalRunsComparison = true;

jest.mock('./hooks/useCompareToRunUuid', () => ({
  useCompareToRunUuid: () => [undefined, mockSetCompareToRunUuid],
}));

jest.mock('./hooks/useRunLoggedTraceTableArtifacts', () => ({
  useRunLoggedTraceTableArtifacts: () => [],
}));

jest.mock('./hooks/useSavePendingEvaluationAssessments', () => ({
  useSavePendingEvaluationAssessments: () => ({}),
}));

jest.mock('./EvaluationRunCompareSelector', () => ({
  EvaluationRunCompareSelector: () => <div data-testid="evaluation-run-compare-selector" />,
}));

jest.mock('../../../common/utils/MarkdownUtils', () => ({
  useMarkdownConverter: () => (markdown?: string) => markdown || '',
}));

jest.mock('../run-page/hooks/useSearchRunsQuery', () => ({
  useSearchRunsQuery: () => ({ data: undefined, loading: false }),
}));

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  shouldEnableImprovedEvalRunsComparison: () => mockShouldEnableImprovedEvalRunsComparison,
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  EXECUTION_DURATION_COLUMN_ID: 'execution_duration',
  GenAiTracesMarkdownConverterProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  GenAiTracesTableDeprecated: () => <div data-testid="artifact-evaluation-table" />,
  STATE_COLUMN_ID: 'state',
  TAGS_COLUMN_ID: 'tags',
  TracesTableColumnType: {
    ASSESSMENT: 'assessment',
    INPUT: 'input',
    TRACE_INFO: 'trace_info',
  },
  useGenAiTraceEvaluationArtifacts: () => ({ data: undefined, isLoading: false }),
}));

const renderComponent = ({
  actions,
  data = [{} as RunEvaluationTracesDataEntry],
}: {
  actions?: React.ReactNode;
  data?: RunEvaluationTracesDataEntry[];
} = {}) =>
  renderWithIntl(
    <DesignSystemProvider>
      <RunViewEvaluationsTabArtifacts
        experimentId="experiment-1"
        runUuid="run-123"
        runDisplayName="Run 123"
        data={data}
        actions={actions}
      />
    </DesignSystemProvider>,
  );

describe('RunViewEvaluationsTabArtifacts', () => {
  beforeEach(() => {
    mockSetCompareToRunUuid.mockClear();
    mockShouldEnableImprovedEvalRunsComparison = true;
  });

  it('renders toolbar actions for artifact-backed evaluation runs', () => {
    renderComponent({ actions: <button type="button">Analyze</button> });

    expect(screen.getByRole('button', { name: 'Analyze' })).toBeInTheDocument();
    expect(screen.getByTestId('artifact-evaluation-table')).toBeInTheDocument();
  });

  it('hides the toolbar when improved comparison is enabled and there are no actions', () => {
    renderComponent();

    expect(screen.queryByTestId('evaluation-run-compare-selector')).not.toBeInTheDocument();
    expect(screen.getByTestId('artifact-evaluation-table')).toBeInTheDocument();
  });

  it('suppresses toolbar actions when no evaluation tables are logged', () => {
    renderComponent({ actions: <button type="button">Analyze</button>, data: [] });

    expect(screen.getByText('No evaluation tables logged')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Analyze' })).not.toBeInTheDocument();
    expect(screen.queryByTestId('artifact-evaluation-table')).not.toBeInTheDocument();
  });

  it('renders the compare selector when improved comparison is disabled', () => {
    mockShouldEnableImprovedEvalRunsComparison = false;

    renderComponent();

    expect(screen.getByTestId('evaluation-run-compare-selector')).toBeInTheDocument();
    expect(screen.getByTestId('artifact-evaluation-table')).toBeInTheDocument();
  });
});
