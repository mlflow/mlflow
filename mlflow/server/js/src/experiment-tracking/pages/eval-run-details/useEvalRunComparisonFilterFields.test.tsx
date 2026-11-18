import { describe, expect, it, jest } from '@jest/globals';
import type { ReactNode } from 'react';
import { renderHook } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { ASSESSMENT_SESSION_METADATA_KEY } from '@databricks/web-shared/model-trace-explorer';
import { FilterOp } from '@databricks/web-shared/traces-table';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { useEvalRunComparisonFilterFields } from './useEvalRunComparisonFilterFields';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

const Wrapper = ({ children }: { children?: ReactNode }) => <IntlProvider locale="en">{children}</IntlProvider>;

describe('useEvalRunComparisonFilterFields', () => {
  it('offers only the fields the comparison can match, suggesting full inputs without changing the traces', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const trace = makeTrace('trace', {
      trace_metadata: { 'mlflow.sourceRun': 'original' },
      assessments: [
        makeFeedbackAssessment('quality', 1),
        makeFeedbackAssessment('session-only', 4, { metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session' } }),
      ],
    });
    const { result } = renderHook(
      () =>
        useEvalRunComparisonFilterFields(
          [{ runUuid: 'selected', traceInfos: [trace] }],
          new Map([['trace', 'full input']]),
        ),
      { wrapper: Wrapper },
    );
    expect(result.current.map(({ id }) => id)).toEqual([
      'state',
      'duration',
      'trace_name',
      'user',
      'session',
      'source',
      'input',
      'output',
      'tag',
      'metadata',
      'assessment',
      'expectation',
    ]);
    expect(result.current.find(({ id }) => id === 'assessment')?.operators).toEqual([
      FilterOp.EQUALS,
      FilterOp.NOT_EQUALS,
    ]);
    expect(result.current.find(({ id }) => id === 'input')?.options).toEqual([
      { value: 'full input', label: 'full input' },
    ]);
    expect(result.current.find(({ id }) => id === 'assessment')?.keyOptions).toEqual([
      { value: 'quality', label: 'quality' },
    ]);
    expect(trace.trace_metadata?.['mlflow.sourceRun']).toBe('original');
    expect(trace.request).toBeUndefined();
  });
});
