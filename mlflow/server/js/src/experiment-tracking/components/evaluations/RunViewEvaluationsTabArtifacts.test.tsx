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

const renderComponent = (actions?: React.ReactNode) =>
  renderWithIntl(
    <DesignSystemProvider>
      <RunViewEvaluationsTabArtifacts
        experimentId="experiment-1"
        runUuid="run-123"
        runDisplayName="Run 123"
        data={[{} as RunEvaluationTracesDataEntry]}
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
    renderComponent(<button type="button">Analyze</button>);

    expect(screen.getByRole('button', { name: 'Analyze' })).toBeInTheDocument();
    expect(screen.getByTestId('artifact-evaluation-table')).toBeInTheDocument();
  });
});
