import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { useExperimentIds } from '../../../components/experiment-page/hooks/useExperimentIds';
import { ScorerEvaluationScope } from '../constants';
import { useRunSerializedScorer } from './useRunSerializedScorer';
import {
  useRunScorerInTracesViewConfiguration,
  useRunScorerInTracesViewConfigurationForExperiment,
} from './useRunScorerInTracesViewConfiguration';

jest.mock('../../../components/experiment-page/hooks/useExperimentIds', () => ({
  useExperimentIds: jest.fn(() => ['route-experiment']),
}));
jest.mock('./useRunSerializedScorer', () => ({
  useRunSerializedScorer: jest.fn(() => ({ evaluateTraces: jest.fn(), allEvaluations: {}, reset: jest.fn() })),
}));

describe('scorer experiment selection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('uses the supplied experiment without reading route state and follows experiment changes', () => {
    const { rerender } = renderHook(({ id }) => useRunScorerInTracesViewConfigurationForExperiment(id), {
      initialProps: { id: 'explicit-experiment' },
    });
    expect(useRunSerializedScorer).toHaveBeenLastCalledWith(
      expect.objectContaining({ experimentId: 'explicit-experiment', scope: ScorerEvaluationScope.TRACES }),
    );
    rerender({ id: 'next-experiment' });
    expect(useRunSerializedScorer).toHaveBeenLastCalledWith(
      expect.objectContaining({ experimentId: 'next-experiment' }),
    );
    expect(useExperimentIds).not.toHaveBeenCalled();
  });

  it('preserves route lookup and session scope for existing callers', () => {
    const { result } = renderHook(() => useRunScorerInTracesViewConfiguration(ScorerEvaluationScope.SESSIONS));
    expect(useExperimentIds).toHaveBeenCalledTimes(1);
    expect(useRunSerializedScorer).toHaveBeenCalledWith(
      expect.objectContaining({ experimentId: 'route-experiment', scope: ScorerEvaluationScope.SESSIONS }),
    );
    expect(result.current.scope).toBe(ScorerEvaluationScope.SESSIONS);
  });
});
