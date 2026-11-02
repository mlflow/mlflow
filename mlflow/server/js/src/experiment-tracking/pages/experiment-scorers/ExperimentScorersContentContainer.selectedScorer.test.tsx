import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from '@databricks/i18n';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestRouter, testRoute } from '../../../common/utils/RoutingTestUtils';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import ExperimentScorersContentContainer from './ExperimentScorersContentContainer';
import { useGetScheduledScorers } from './hooks/useGetScheduledScorers';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));
jest.mock('./hooks/useGetScheduledScorers', () => ({ useGetScheduledScorers: jest.fn() }));
jest.mock('./ScorerCardContainer', () => ({
  __esModule: true,
  default: ({ scorer, initiallyExpanded }: { scorer: { name: string }; initiallyExpanded?: boolean }) => (
    <div data-testid={`scorer-card-${scorer.name}`}>{initiallyExpanded === undefined ? 'default' : 'expanded'}</div>
  ),
}));
jest.mock('./ScorerModalRenderer', () => ({ __esModule: true, default: () => null }));

const renderPage = (url: string, scorerNames = ['quality', 'tone']) => {
  jest.mocked(useGetScheduledScorers).mockReturnValue({
    data: { scheduledScorers: scorerNames.map((name) => ({ name })) },
    isLoading: false,
    isError: false,
    error: null,
  } as unknown as ReturnType<typeof useGetScheduledScorers>);
  render(
    <TestRouter
      initialEntries={[url]}
      routes={[
        testRoute(
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <ExperimentScorersContentContainer experimentId="exp-1" />
            </DesignSystemProvider>
          </IntlProvider>,
          '/experiments/:experimentId/judges',
        ),
      ]}
    />,
  );
};

describe('ExperimentScorersContentContainer linked scorer', () => {
  test('flag on: opens the scorer named in the link', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage('/experiments/exp-1/judges?selectedScorerName=quality');

    expect(screen.getByTestId('scorer-card-quality')).toHaveTextContent('expanded');
    expect(screen.getByTestId('scorer-card-tone')).toHaveTextContent('default');
    expect(screen.queryByText(/isn't registered in this experiment/)).not.toBeInTheDocument();
  });

  test("flag on: says so when the linked scorer isn't registered in this experiment", () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage('/experiments/exp-1/judges?selectedScorerName=safety');

    expect(screen.getByText("The judge “safety” isn't registered in this experiment.")).toBeInTheDocument();
    expect(screen.getByTestId('scorer-card-quality')).toHaveTextContent('default');
  });

  test('flag on: says so above the empty state when the experiment has no judges', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage('/experiments/exp-1/judges?selectedScorerName=safety', []);

    expect(screen.getByText("The judge “safety” isn't registered in this experiment.")).toBeInTheDocument();
  });

  test('flag off: ignores the link and leaves every card as before', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderPage('/experiments/exp-1/judges?selectedScorerName=quality');

    expect(screen.getByTestId('scorer-card-quality')).toHaveTextContent('default');
    expect(screen.getByTestId('scorer-card-tone')).toHaveTextContent('default');
  });

  test('flag off: shows no message for a scorer the experiment does not have', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderPage('/experiments/exp-1/judges?selectedScorerName=safety');

    expect(screen.queryByText(/isn't registered in this experiment/)).not.toBeInTheDocument();
  });
});
