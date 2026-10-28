import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import {
  TRACE_DRAWER_SESSION_ID_QUERY_PARAM,
  TRACE_DRAWER_VIEW_MODE_QUERY_PARAM,
} from '@mlflow/mlflow/src/experiment-tracking/constants';

const TRACE_ID_PARAM = 'traceId';
const LEGACY_TRACE_ID_PARAM = 'selectedEvaluationId';
const SPAN_ID_PARAM = 'spanId';
const SESSION_VIEW_MODE = 'session';

export type TracesV4DrawerViewMode = 'trace' | 'session';

export interface TracesV4DrawerSessionUrlState {
  /** The session the drawer was opened from, kept while one of its traces is shown. */
  sessionId: string | undefined;
  viewMode: TracesV4DrawerViewMode;
  /** Opens the drawer on a session's conversation, with `traceId` as the selected turn. */
  openSession: (sessionId: string, traceId: string) => void;
  /** Shows one trace of the open session in the drawer, keeping the session to go back to. */
  showSessionTrace: (traceId: string) => void;
  /** Returns from a trace to its session's conversation. */
  showSession: () => void;
  /** Opens a trace on its own (for example the next trace on the page), leaving the session. */
  selectTrace: (traceId: string) => void;
  /** Closes the drawer and forgets the session. */
  closeDrawer: () => void;
}

/**
 * URL state for opening sessions in the V4 trace drawer: `traceDrawerView=session` and
 * `traceDrawerSessionId=<id>`, written next to the table's own parameters so its filters, sort and
 * grouping stay as they are.
 */
export const useTracesV4DrawerSessionUrlState = (): TracesV4DrawerSessionUrlState => {
  const [searchParams, setSearchParams] = useSearchParams();
  const sessionId = searchParams.get(TRACE_DRAWER_SESSION_ID_QUERY_PARAM) ?? undefined;
  const viewMode: TracesV4DrawerViewMode =
    sessionId && searchParams.get(TRACE_DRAWER_VIEW_MODE_QUERY_PARAM) === SESSION_VIEW_MODE ? 'session' : 'trace';

  const openSession = useCallback(
    (nextSessionId: string, traceId: string) => {
      setSearchParams((params) => {
        params.delete(LEGACY_TRACE_ID_PARAM);
        params.delete(SPAN_ID_PARAM);
        params.set(TRACE_ID_PARAM, traceId);
        params.set(TRACE_DRAWER_VIEW_MODE_QUERY_PARAM, SESSION_VIEW_MODE);
        params.set(TRACE_DRAWER_SESSION_ID_QUERY_PARAM, nextSessionId);
        return params;
      });
    },
    [setSearchParams],
  );

  const showSessionTrace = useCallback(
    (traceId: string) => {
      setSearchParams((params) => {
        params.delete(LEGACY_TRACE_ID_PARAM);
        params.delete(SPAN_ID_PARAM);
        params.delete(TRACE_DRAWER_VIEW_MODE_QUERY_PARAM);
        params.set(TRACE_ID_PARAM, traceId);
        return params;
      });
    },
    [setSearchParams],
  );

  const showSession = useCallback(() => {
    setSearchParams((params) => {
      params.delete(SPAN_ID_PARAM);
      params.set(TRACE_DRAWER_VIEW_MODE_QUERY_PARAM, SESSION_VIEW_MODE);
      return params;
    });
  }, [setSearchParams]);

  const selectTrace = useCallback(
    (traceId: string) => {
      setSearchParams((params) => {
        params.delete(LEGACY_TRACE_ID_PARAM);
        params.delete(SPAN_ID_PARAM);
        params.delete(TRACE_DRAWER_VIEW_MODE_QUERY_PARAM);
        params.delete(TRACE_DRAWER_SESSION_ID_QUERY_PARAM);
        params.set(TRACE_ID_PARAM, traceId);
        return params;
      });
    },
    [setSearchParams],
  );

  const closeDrawer = useCallback(() => {
    setSearchParams((params) => {
      params.delete(LEGACY_TRACE_ID_PARAM);
      params.delete(TRACE_ID_PARAM);
      params.delete(SPAN_ID_PARAM);
      params.delete(TRACE_DRAWER_VIEW_MODE_QUERY_PARAM);
      params.delete(TRACE_DRAWER_SESSION_ID_QUERY_PARAM);
      return params;
    });
  }, [setSearchParams]);

  return { sessionId, viewMode, openSession, showSessionTrace, showSession, selectTrace, closeDrawer };
};
