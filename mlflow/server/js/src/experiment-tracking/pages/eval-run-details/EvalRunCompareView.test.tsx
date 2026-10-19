import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { DesignSystemProvider } from '@databricks/design-system';
import { EvalRunCompareView } from './EvalRunCompareView';

jest.mock('./useEvalRunComparisonBaseline', () => ({ useEvalRunComparisonBaseline: () => jest.fn() }));
jest.mock('./EvalRunCompareSelectionBar', () => ({
  EvalRunCompareSelectionBar: () => <div data-testid="selection-bar" />,
}));
jest.mock('./EvalRunComparisonOverviewTab', () => ({
  EvalRunComparisonOverviewTab: ({
    experimentId,
    baselineRunUuid,
    compareRunUuids,
    onNavigateToTraceView,
  }: {
    experimentId: string;
    baselineRunUuid: string;
    compareRunUuids: string[];
    onNavigateToTraceView?: () => void;
  }) => (
    <div data-testid="overview">
      {experimentId} base:{baselineRunUuid} compare:{compareRunUuids.join(',')} links:
      {onNavigateToTraceView ? 'on' : 'off'}
    </div>
  ),
}));

describe('EvalRunCompareView', () => {
  test('shows the comparison overview of the selected runs under the selection bar', () => {
    render(
      <DesignSystemProvider>
        <EvalRunCompareView experimentId="exp-1" baselineRunUuid="run-1" compareRunUuids={['run-2', 'run-3']} />
      </DesignSystemProvider>,
    );
    expect(screen.getByTestId('selection-bar')).toBeInTheDocument();
    expect(screen.getByTestId('overview')).toHaveTextContent('exp-1 base:run-1 compare:run-2,run-3 links:off');
    expect(
      screen.getByTestId('selection-bar').compareDocumentPosition(screen.getByTestId('overview')) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
