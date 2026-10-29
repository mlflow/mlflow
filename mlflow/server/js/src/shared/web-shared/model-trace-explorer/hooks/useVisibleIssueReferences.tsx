import {
  SEARCH_ISSUES_QUERY_KEY,
  searchIssueIdsByStatus,
  type IssueStatus,
} from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery';

import { useQuery } from '../../query-client/queryClient';

import { useSafeSelectedIssueId } from './useSafeSelectedIssueId';
import {
  getUniqueIssueReferenceAssessments,
  getUniqueIssueReferences,
  type TraceIssueReference,
} from '../IssueAssessmentUtils';
import type { Assessment, IssueReferenceAssessment } from '../ModelTrace.types';
import { useParams } from '../RoutingUtils';

/** The status the Issues tab labels "Not an issue". `resolved` is not dismissed: it did happen. */
const DISMISSED_ISSUE_STATUS: IssueStatus = 'rejected';

/** Status is changed by hand and rarely, and paging through traces re-reads the same experiment. */
const DISMISSED_ISSUE_IDS_STALE_TIME_MS = 5 * 60 * 1000;

const NO_DISMISSED_ISSUE_IDS: ReadonlySet<string> = new Set();

/**
 * Every issue in the experiment that someone dismissed, by ID.
 *
 * One request per experiment, not one per issue on screen: the span tree, the span's cards, the
 * assessments pane and the traces table share this entry, so a table page naming forty issues reads
 * once. A trace referencing no issue asks for nothing.
 *
 * Keyed under `SEARCH_ISSUES_QUERY_KEY` so that triaging an issue invalidates this alongside the
 * Issues tab's own reads.
 *
 * The experiment comes from the route, as it does elsewhere in this package. Off an experiment
 * route there is nothing to look up, so nothing is hidden -- the same outcome as a lookup that
 * fails, since failing to reach the issues API is not evidence that anyone dismissed anything.
 */
const useDismissedIssueIds = (enabled: boolean): ReadonlySet<string> => {
  const { experimentId } = useParams();
  const { data } = useQuery({
    queryKey: [SEARCH_ISSUES_QUERY_KEY, 'dismissed', experimentId],
    queryFn: async () => new Set(await searchIssueIdsByStatus(experimentId ?? '', DISMISSED_ISSUE_STATUS)),
    enabled: enabled && Boolean(experimentId),
    refetchOnWindowFocus: false,
    staleTime: DISMISSED_ISSUE_IDS_STALE_TIME_MS,
    retry: false,
  });

  return data ?? NO_DISMISSED_ISSUE_IDS;
};

/**
 * Which of a span's referenced issues are still worth showing, by issue ID.
 *
 * A trace keeps its issue references forever -- marking an issue "Not an issue" does not retract
 * the reference detection wrote -- and the status lives on the issue rather than on the reference,
 * so it has to be read separately.
 *
 * An issue named in the URL is kept even when dismissed: arriving from that issue's own page is how
 * a reader reviews the evidence behind it. Issues also show until the dismissed set arrives rather
 * than being withheld until it does, so a trace whose issues are all open is never held back.
 */
export const useVisibleIssueIds = (issueIds: string[]): ReadonlySet<string> => {
  const selectedIssueId = useSafeSelectedIssueId();
  const dismissedIssueIds = useDismissedIssueIds(issueIds.length > 0);

  return new Set(issueIds.filter((issueId) => issueId === selectedIssueId || !dismissedIssueIds.has(issueId)));
};

/** The issues a span is flagged for that nobody has dismissed, as the cards need them. */
export const useVisibleIssueReferences = (assessments: Assessment[] | undefined): TraceIssueReference[] => {
  const references = getUniqueIssueReferences(assessments ?? []);
  const visibleIssueIds = useVisibleIssueIds(references.map(({ id }) => id));
  return references.filter(({ id }) => visibleIssueIds.has(id));
};

/** The same selection, as the assessments they came from. */
export const useVisibleIssueReferenceAssessments = (
  assessments: IssueReferenceAssessment[],
): IssueReferenceAssessment[] => {
  const references = getUniqueIssueReferenceAssessments(assessments);
  const visibleIssueIds = useVisibleIssueIds(references.map(({ assessment_name }) => assessment_name));
  return references.filter(({ assessment_name }) => visibleIssueIds.has(assessment_name));
};
