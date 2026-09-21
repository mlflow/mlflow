import { describe, expect, it, jest } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { IntlProvider } from '@databricks/i18n';
import { useRunSerializedScorer } from './useRunSerializedScorer';
import { useEvaluateTraces } from '../useEvaluateTraces';

jest.mock('../useEvaluateTraces');

const wrapper = ({ children }: { children: React.ReactNode }) => <IntlProvider locale="en">{children}</IntlProvider>;

describe('useRunSerializedScorer', () => {
  it('runs a saved Jev scorer on traces using its native configuration', async () => {
    const evaluate = jest.fn<ReturnType<typeof useEvaluateTraces>[0]>().mockResolvedValue(undefined);
    jest.mocked(useEvaluateTraces).mockReturnValue([
      evaluate,
      {
        latestEvaluation: null,
        isLoading: false,
        error: null,
        reset: jest.fn(),
        allEvaluations: {},
      },
    ] as ReturnType<typeof useEvaluateTraces>);
    const { result } = renderHook(() => useRunSerializedScorer({ experimentId: '1' }), { wrapper });
    await act(async () =>
      result.current.evaluateTraces(
        {
          type: 'jev',
          name: 'quality',
          model: 'gateway:/jev',
          question: 'Is it correct?',
          answerType: 'noul',
          criteria: null,
          threshold: 0.7,
        },
        ['tr-1'],
      ),
    );
    expect(evaluate.mock.calls[0][0]).toMatchObject({ itemIds: ['tr-1'], saveAssessment: true });
    expect(JSON.parse(evaluate.mock.calls[0][0].serializedScorer!)).toMatchObject({
      jev_scorer_pydantic_data: { model: 'gateway:/jev', answer_type: 'noul', threshold: 0.7 },
    });
  });
});
