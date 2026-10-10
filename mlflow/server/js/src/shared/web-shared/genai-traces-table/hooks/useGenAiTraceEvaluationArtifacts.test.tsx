import { describe, expect, it, jest } from '@jest/globals';
import { getArtifactChunkedText } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import { fetchRunArtifactWithPresignedUrl } from '@mlflow/mlflow/src/experiment-tracking/utils/PresignedArtifactUtils';

import { GenAiTraceEvaluationArtifactFile } from '../enum';
import { fetchGenAiTraceEvaluationArtifact } from './useGenAiTraceEvaluationArtifacts';

jest.mock('@mlflow/mlflow/src/experiment-tracking/utils/PresignedArtifactUtils', () => ({
  fetchRunArtifactWithPresignedUrl: jest.fn(),
}));

describe('fetchGenAiTraceEvaluationArtifact', () => {
  it('reads trace evaluation tables through the presigned artifact resolver', async () => {
    jest
      .mocked(fetchRunArtifactWithPresignedUrl)
      .mockResolvedValue(JSON.stringify({ columns: ['evaluation_id'], data: [['eval-1']] }));

    await expect(
      fetchGenAiTraceEvaluationArtifact('run-123', GenAiTraceEvaluationArtifactFile.Evaluations),
    ).resolves.toEqual({
      columns: ['evaluation_id'],
      data: [['eval-1']],
      filename: GenAiTraceEvaluationArtifactFile.Evaluations,
    });
    expect(fetchRunArtifactWithPresignedUrl).toHaveBeenCalledWith(
      'run-123',
      GenAiTraceEvaluationArtifactFile.Evaluations,
      expect.stringContaining('run_uuid=run-123'),
      getArtifactChunkedText,
    );
  });
});
