import { useCallback } from 'react';
import { useSearchParams } from '../../../common/utils/RoutingUtils';
import {
  useTracesV4UrlState,
  type TracesV4UrlState,
} from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4UrlState';

const LEGACY_TRACE_ID_PARAM = 'selectedEvaluationId';
const TRACE_ID_PARAM = 'traceId';
const SPAN_ID_PARAM = 'spanId';

/**
 * Eval-run URL state keeps the legacy drawer deep link canonical while accepting links copied from
 * the experiment V4 table. All other paging, search, sort, and tag state follows the shared V4 URL
 * contract.
 */
export const useEvalRunTracesV4UrlState = (): TracesV4UrlState => {
  const base = useTracesV4UrlState();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedEvaluationId = searchParams.get(LEGACY_TRACE_ID_PARAM) ?? undefined;

  const setTraceId = useCallback(
    (next: string | undefined) => {
      setSearchParams((params) => {
        if (next) {
          params.set(LEGACY_TRACE_ID_PARAM, next);
        } else {
          params.delete(LEGACY_TRACE_ID_PARAM);
        }
        params.delete(TRACE_ID_PARAM);
        params.delete(SPAN_ID_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  return {
    ...base,
    traceId: selectedEvaluationId ?? base.traceId,
    setTraceId,
  };
};
