import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, useLocation } from '../../../../common/utils/RoutingUtils';
import { renderWithDesignSystem } from '../../../../common/utils/TestUtils.react18';
import type { GenAIOverviewEvalScorePoint } from '../hooks/useGenAIOverviewEvalState';
import { GenAIOverviewEvalScoresChart, type GenAIOverviewEvalScoresChartProps } from './GenAIOverviewEvalScoresChart';

const CurrentTestLocation = () => {
  const { pathname } = useLocation();
  return (
    <div role="status" aria-label="Current test location">
      {pathname}
    </div>
  );
};

describe('GenAIOverviewEvalScoresChart', () => {
  test('limits Assistant context to ten scores while preserving the total count', async () => {
    const onAskAssistant = jest.fn<NonNullable<GenAIOverviewEvalScoresChartProps['onAskAssistant']>>();
    const scoreNames = Array.from({ length: 11 }, (_, index) => `score-${index + 1}`);
    const scores = Object.fromEntries(scoreNames.map((scoreName, index) => [scoreName, index / 20]));
    const scorePoints: GenAIOverviewEvalScorePoint[] = [
      { runUuid: 'run-1', runName: 'candidate', timestampMs: 100, scores },
    ];
    renderWithDesignSystem(
      <MemoryRouter>
        <GenAIOverviewEvalScoresChart
          experimentId="experiment-1"
          assessmentScoreNames={scoreNames}
          scorePoints={scorePoints}
          onAskAssistant={onAskAssistant}
        />
      </MemoryRouter>,
    );

    fireEvent.keyDown(screen.getByRole('button', { name: /Evaluation score trends/ }), { key: 'Enter' });
    await userEvent.click(await screen.findByRole('button', { name: 'Ask Assistant' }));

    expect(onAskAssistant).toHaveBeenCalledWith(
      expect.objectContaining({
        stage: 'eval',
        totalScoreCount: 11,
        runs: [
          expect.objectContaining({
            scores: Object.fromEntries(scoreNames.slice(0, 10).map((scoreName) => [scoreName, scores[scoreName]])),
          }),
        ],
      }),
    );
  });
  test("opens the selected point's own evaluation run, not the experiment Issues tab", async () => {
    // Every point on this chart is an evaluation run, so the selected run id is the whole answer:
    // sending it to a shared destination would discard what the user picked.
    const scorePoints: GenAIOverviewEvalScorePoint[] = [
      { runUuid: 'run-7', runName: 'candidate', timestampMs: 100, scores: { correctness: 0.5 } },
    ];
    renderWithDesignSystem(
      <MemoryRouter>
        <GenAIOverviewEvalScoresChart
          experimentId="experiment-1"
          assessmentScoreNames={['correctness']}
          scorePoints={scorePoints}
        />
        <CurrentTestLocation />
      </MemoryRouter>,
    );

    fireEvent.keyDown(screen.getByRole('button', { name: /Evaluation score trends/ }), { key: 'Enter' });
    await userEvent.click(await screen.findByRole('button', { name: 'Open eval run' }));

    expect(screen.getByRole('status', { name: 'Current test location' })).toHaveTextContent(
      '/experiments/experiment-1/evaluation-runs/run-7/details',
    );
  });
});
