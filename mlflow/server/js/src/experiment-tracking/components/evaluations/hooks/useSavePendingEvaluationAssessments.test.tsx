import { act, renderHook, waitFor } from '@testing-library/react';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { uploadArtifactApi } from '../../../actions';
import { useSavePendingEvaluationAssessments } from './useSavePendingEvaluationAssessments';
import type {
  RunEvaluationResultAssessment,
  RunEvaluationResultAssessmentDraft,
} from '@databricks/web-shared/genai-traces-table';
import { getArtifactChunkedText } from '../../../../common/utils/ArtifactUtils';
import { merge } from 'lodash';
import { KnownEvaluationResultAssessmentMetadataFields } from '@databricks/web-shared/genai-traces-table';

jest.mock('../../../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/ArtifactUtils')>(
    '../../../../common/utils/ArtifactUtils',
  ),
  getArtifactChunkedText: jest.fn(),
}));

jest.mock('../../../actions', () => ({
  getEvaluationTableArtifact: jest.fn(),
  uploadArtifactApi: jest.fn(),
}));

const sampleExistingAssessmentSourceMetadata = {
  key_a: '1',
  key_b: '2',
};

const sampleExistingAssessmentData = {
  columns: [
    'evaluation_id',
    'name',
    'source',
    'timestamp',
    'boolean_value',
    'numeric_value',
    'string_value',
    'rationale',
    'metadata',
  ],
  data: [
    [
      'test-evaluation-id',
      'overall_assessment',
      { source_id: '', source_metadata: {}, source_type: 'AI_JUDGE' },
      1,
      null,
      null,
      'some-existing-assessment-value',
      '',
      { is_overall_assessment: true },
    ],
    [
      'test-evaluation-id',
      'groundedness',
      { source_id: '', source_metadata: {}, source_type: 'AI_JUDGE' },
      100,
      null,
      null,
      'no',
      '',
      {},
    ],
    [
      'test-evaluation-id',
      'groundedness',
      { source_id: 'user@user.com', source_metadata: sampleExistingAssessmentSourceMetadata, source_type: 'HUMAN' },
      200,
      null,
      null,
      'yes',
      '',
      {},
    ],
    [
      'unrelated-evaluation-id',
      'overall_assessment',
      { source_id: '', source_metadata: {}, source_type: 'AI_JUDGE' },
      1,
      null,
      null,
      'some-existing-assessment-value',
      '',
      { is_overall_assessment: true },
    ],
  ],
};

describe('useSavePendingEvaluationAssessments', () => {
  const createDeferredUploadArtifactApi = () => {
    const result: { resolveUpload: () => void } = { resolveUpload: jest.fn() };
    const deferredPromise = new Promise<void>((resolve) => {
      result.resolveUpload = resolve;
    });
    jest
      .mocked(uploadArtifactApi)
      .mockImplementation(() => ({ type: 'uploadArtifactApi', payload: deferredPromise } as any));

    return result;
  };

  beforeEach(() => {
    jest
      .mocked(uploadArtifactApi)
      .mockImplementation(() => ({ type: 'uploadArtifactApi', payload: Promise.resolve() } as any));
    jest.mocked(getArtifactChunkedText).mockImplementation(() => {
      return Promise.resolve(JSON.stringify(sampleExistingAssessmentData));
    });
  });
  const renderTestHook = () =>
    renderHook(() => useSavePendingEvaluationAssessments(), {
      wrapper: MockedReduxStoreProvider,
    });

  const createAssessmentObject = (
    name: string,
    value: string,
    isHuman = false,
    isDraft = false,
    isOverall = false,
    sourceId = '',
  ): RunEvaluationResultAssessment | RunEvaluationResultAssessmentDraft => ({
    name,
    stringValue: value,
    booleanValue: null,
    metadata: isOverall ? { [KnownEvaluationResultAssessmentMetadataFields.IS_OVERALL_ASSESSMENT]: true } : {},
    numericValue: null,
    rationale: '',
    source: {
      metadata: {},
      sourceId,
      sourceType: isHuman ? 'HUMAN' : 'AI_JUDGE',
    },
    timestamp: 0,
    isDraft: isDraft || undefined,
  });

  test('it should save pending assessments to the simulated upload function', async () => {
    const sampleEvaluationAssessments = [
      createAssessmentObject('overall_assessment', 'yes', true, false, true),
      createAssessmentObject('overall_assessment', 'weak yes', true, false, true),
      createAssessmentObject('overall_assessment', 'no', false, false, true),
      createAssessmentObject('correctness', 'no', true, true),
      createAssessmentObject('correctness', 'yes', false),
    ] as RunEvaluationResultAssessmentDraft[];

    const { resolveUpload } = createDeferredUploadArtifactApi();

    const { result } = renderTestHook();

    await act(async () => {
      result.current.savePendingAssessments('test-run-uuid', 'test-evaluation-id', sampleEvaluationAssessments);
    });

    await waitFor(() => {
      expect(result.current.isSaving).toBe(true);
    });

    resolveUpload();

    await waitFor(() => {
      expect(result.current.isSaving).toBe(false);
    });
    const [sentTestRunUuid, sentArtifactPath, sentContents] = jest.mocked(uploadArtifactApi).mock.lastCall || [];

    expect(sentTestRunUuid).toEqual('test-run-uuid');
    expect(sentArtifactPath).toEqual('_assessments.json');
    expect(sentContents.columns).toEqual(sampleExistingAssessmentData.columns);

    const originalUnrelatedEvaluationDataEntry = sampleExistingAssessmentData.data.find(
      ([evaluationId]) => evaluationId === 'unrelated-evaluation-id',
    );

    expect(sentContents.data).toContainEqual(originalUnrelatedEvaluationDataEntry);

    expect(sentContents.data).toContainEqual(expect.arrayContaining(['overall_assessment', 'yes']));
    expect(sentContents.data).toContainEqual(expect.arrayContaining(['overall_assessment', 'weak yes']));
    expect(sentContents.data).toContainEqual(expect.arrayContaining(['overall_assessment', 'no']));

    expect(sentContents.data).toContainEqual(expect.arrayContaining(['correctness', 'no']));
    expect(sentContents.data).toContainEqual(expect.arrayContaining(['correctness', 'yes']));
  });

  describe('it should discard existing entries when incoming assessment have matching sources', () => {
    // Create a new assessment with the same name as an existing assessment
    const userProvidedTestAssessment = createAssessmentObject(
      'groundedness',
      'no',
      true,
      true,
      false,
      'user@user.com',
    ) as RunEvaluationResultAssessmentDraft;

    const assessmentWithSameSource = merge({}, userProvidedTestAssessment, {
      source: { metadata: sampleExistingAssessmentSourceMetadata },
    });
    const assessmentWithEqualSource = merge({}, userProvidedTestAssessment, {
      // Change order of keys in assessment
      name: userProvidedTestAssessment.name,
      // Change order of keys in source metadata
      source: { metadata: { key_b: '2', key_a: '1' } },
    });

    test.each([
      ['same', assessmentWithSameSource],
      ['value-equal', assessmentWithEqualSource],
    ])('For %s assessment object', async (name, userAssessment) => {
      const { resolveUpload } = createDeferredUploadArtifactApi();
      const { result } = renderTestHook();

      await act(async () => {
        result.current.savePendingAssessments('test-run-uuid', 'test-evaluation-id', [userAssessment]);
      });

      resolveUpload();

      await waitFor(() => {
        expect(result.current.isSaving).toBe(false);
      });

      const [, , sentContents] = jest.mocked(uploadArtifactApi).mock.lastCall || [];

      // The existing assessment with the same source should be removed
      expect(sentContents.data).not.toContainEqual(
        expect.arrayContaining(['groundedness', expect.objectContaining({ source_id: 'user@user.com' }), 'yes']),
      );
    });
  });
});
