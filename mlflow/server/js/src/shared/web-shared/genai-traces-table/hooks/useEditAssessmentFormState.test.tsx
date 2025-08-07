import { renderHook, act } from '@testing-library/react';

import { IntlProvider } from '@databricks/i18n';

import { useEditAssessmentFormState } from './useEditAssessmentFormState';
import {
  KnownEvaluationResponseAssessmentNames,
  KnownEvaluationResultAssessmentName,
} from '../components/GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo } from '../types';

const KNOWN_ASSESSMENT_INFOS = KnownEvaluationResponseAssessmentNames.map((name) => getKnownAssessmentInfos(name));

function getKnownAssessmentInfos(name: string): AssessmentInfo {
  return {
    name,
    displayName: name,
    isKnown: true,
    isOverall: name === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
    metricName: 'test-metric',
    source: {
      sourceType: 'AI_JUDGE',
      sourceId: 'test-source-id',
      metadata: {},
    },
    isCustomMetric: false,
    isEditable: true,
    isRetrievalAssessment: false,
    dtype: 'string',
    uniqueValues: new Set(['yes', 'no']),
    docsLink: 'https://docs.example.com',
    missingTooltip: 'Missing tooltip',
    description: 'Test assessment description',
  };
}

describe('useEditAssessmentFormState', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  const renderTestedHook = (assessmentInfos?: AssessmentInfo[]) =>
    renderHook(() => useEditAssessmentFormState([], assessmentInfos), {
      wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
    });

  it('should initialize with correct initial state', () => {
    const { result } = renderTestedHook([]);

    expect(result.current.suggestions).toEqual([]);
    expect(result.current.editingAssessment).toBeUndefined();
    expect(result.current.showUpsertForm).toBeFalsy();
  });

  it('should be empty list for known assessment infos when adding new tag', () => {
    const { result } = renderTestedHook(KNOWN_ASSESSMENT_INFOS);

    act(() => {
      result.current.addAssessment();
    });

    expect(result.current.showUpsertForm).toBeTruthy();
    expect(result.current.editingAssessment).toBeUndefined();

    expect(result.current.suggestions).toEqual([]);
  });

  it('should pre-populate boolean assessments', () => {
    const humanTagAssessment: AssessmentInfo = {
      name: 'custom-assessment',
      displayName: 'Custom Assessment',
      isKnown: false,
      isOverall: false,
      metricName: 'test-metric',
      source: {
        sourceType: 'HUMAN',
        sourceId: 'test-source-id',
        metadata: {},
      },
      isCustomMetric: false,
      isEditable: true,
      isRetrievalAssessment: false,
      dtype: 'boolean',
      uniqueValues: new Set([true, false]),
      docsLink: 'https://docs.example.com',
      missingTooltip: 'Missing tooltip',
      description: 'Test assessment description',
    };

    const { result } = renderTestedHook([...KNOWN_ASSESSMENT_INFOS, humanTagAssessment]);

    act(() => {
      result.current.addAssessment();
    });

    expect(result.current.showUpsertForm).toBeTruthy();
    expect(result.current.editingAssessment).toBeUndefined();

    expect(result.current.suggestions).toEqual([
      {
        key: humanTagAssessment.name,
        label: humanTagAssessment.name,
        rootAssessmentName: 'custom-assessment',
        disabled: false,
      },
    ]);
  });

  it('should set form state with original assessment for editing', () => {
    const originalAssessment = {
      name: KnownEvaluationResultAssessmentName.CORRECTNESS,
      stringValue: 'yes',
    } as any;
    const { result } = renderTestedHook(KNOWN_ASSESSMENT_INFOS);

    act(() => {
      result.current.editAssessment(originalAssessment);
    });

    expect(result.current.showUpsertForm).toBeTruthy();
    expect(result.current.editingAssessment).toEqual(originalAssessment);
    expect(result.current.suggestions).toEqual(
      [
        ['yes', 'Correct', KnownEvaluationResultAssessmentName.CORRECTNESS],
        ['no', 'Incorrect', KnownEvaluationResultAssessmentName.CORRECTNESS],
      ].map(([key, label, rootAssessmentName]) => ({ key, label, rootAssessmentName })),
    );
  });

  it('should cancel edit and reset form state', () => {
    const { result } = renderTestedHook(KNOWN_ASSESSMENT_INFOS);

    act(() => {
      result.current.addAssessment();
    });

    expect(result.current.showUpsertForm).toBeTruthy();

    act(() => {
      result.current.closeForm();
    });

    expect(result.current.showUpsertForm).toBeFalsy();
    expect(result.current.editingAssessment).toBeUndefined();
  });
});
