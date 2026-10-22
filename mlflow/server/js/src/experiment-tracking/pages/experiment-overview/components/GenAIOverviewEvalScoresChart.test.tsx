import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { renderWithDesignSystem } from '../../../../common/utils/TestUtils.react18';
import type { GenAIOverviewEvalScorePoint } from '../hooks/useGenAIOverviewEvalState';
import { GenAIOverviewEvalScoresChart, type GenAIOverviewEvalScoresChartProps } from './GenAIOverviewEvalScoresChart';

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
});
