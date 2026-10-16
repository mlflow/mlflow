import { Alert, Button, TableSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { compact } from 'lodash';
import { getIssueReferenceFromAssessment, useGetTracesById } from '../../../shared/web-shared/model-trace-explorer';
import type { ModelTraceInfoV3, TraceIssueReference } from '../../../shared/web-shared/model-trace-explorer';
import {
  toExplorerSpanId,
  toIssueEvidenceDetail,
  toIssueEvidenceHeader,
  toSpanIdentities,
  type IssueEvidenceDetail,
  type IssueEvidenceHeader,
} from './issueEvidence.utils';
import { IssueEvidenceCard } from './IssueEvidenceCard';
import { IssueDetailSectionTitle } from './IssueDetailSectionTitle';
import { IssueTraceDrawer } from './IssueTraceDrawer';
import { ISSUE_EVIDENCE_PAGE_SIZE, useIssueEvidenceTraces } from './useIssueEvidenceTraces';

export interface IssueEvidenceSectionProps {
  experimentId: string;
  issueId: string;
  /**
   * Opens a trace in the drawer this section hosts, on `spanId` where the card knows which span it
   * is sending the reader to.
   */
  onViewTrace: (linkableTraceId: string, spanId?: string) => void;
}

/**
 * The newest valid reference to this issue on a trace. Repeated detection runs can leave several,
 * and an invalidated one is not an association at all.
 */
const getIssueReference = (info: ModelTraceInfoV3, issueId: string): TraceIssueReference | undefined => {
  const references = compact(
    (info.assessments ?? []).map((assessment) => {
      if (assessment.valid === false) {
        return undefined;
      }
      const reference = getIssueReferenceFromAssessment(assessment);
      if (reference?.id !== issueId) {
        return undefined;
      }
      const lastUpdateTime = Date.parse(assessment.last_update_time ?? assessment.create_time);
      return {
        assessmentId: assessment.assessment_id,
        lastUpdateTime: Number.isNaN(lastUpdateTime) ? 0 : lastUpdateTime,
        reference,
      };
    }),
  );
  references.sort(
    (left, right) => right.lastUpdateTime - left.lastUpdateTime || right.assessmentId.localeCompare(left.assessmentId),
  );
  return references[0]?.reference;
};

/**
 * The failure evidence behind an issue, one card per affected trace.
 *
 * Reads in two stages because a card's halves differ in cost by orders of magnitude: every header
 * comes from the paged trace-info search that also establishes membership, while an expanded
 * card's spans are read only for that card.
 */
export const IssueEvidenceSection = ({ experimentId, issueId, onViewTrace }: IssueEvidenceSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [expandedTraceIds, setExpandedTraceIds] = useState<string[] | undefined>(undefined);
  const traces = useIssueEvidenceTraces(experimentId, issueId);

  // One header per affected trace, already in the search's order.
  const headers = useMemo(() => compact(traces.traceInfos.map(toIssueEvidenceHeader)), [traces.traceInfos]);

  const issueReferenceByTraceId = useMemo(() => {
    const references = new Map<string, TraceIssueReference>();
    for (const info of traces.traceInfos) {
      const reference = getIssueReference(info, issueId);
      if (reference) {
        references.set(info.trace_id, reference);
      }
    }
    return references;
  }, [traces.traceInfos, issueId]);

  // Until the user touches a card the leading one is open -- the common case is one issue, one
  // representative failure, and making them click to see any evidence at all defeats the point.
  const isExpanded = useCallback(
    (traceId: string, index: number) => (expandedTraceIds ? expandedTraceIds.includes(traceId) : index === 0),
    [expandedTraceIds],
  );

  // A reference with structured evidence needs spans to name the ones its steps cite; one with
  // only a rationale needs them to describe the span it blamed. A reference with neither needs no
  // read at all.
  const needsSpans = useCallback(
    (traceId: string) => {
      const reference = issueReferenceByTraceId.get(traceId);
      return Boolean(reference?.evidence?.steps.length || reference?.spanId);
    },
    [issueReferenceByTraceId],
  );

  const spansTraceIdsToRead = useMemo(
    () =>
      headers
        .filter((header, index) => isExpanded(header.traceId, index) && needsSpans(header.traceId))
        .map((header) => header.addressableTraceId),
    [headers, isExpanded, needsSpans],
  );

  const details = useGetTracesById(spansTraceIdsToRead);

  const fetchedTraceByTraceId = useMemo(
    () =>
      new Map(
        compact(
          details.data.map((trace) =>
            trace.info && 'trace_id' in trace.info ? ([trace.info.trace_id, trace] as const) : undefined,
          ),
        ),
      ),
    [details.data],
  );

  const spansByTraceId = useMemo(
    () =>
      new Map(
        headers.map((header) => [header.traceId, toSpanIdentities(fetchedTraceByTraceId.get(header.traceId))] as const),
      ),
    [fetchedTraceByTraceId, headers],
  );

  // What a rationale-only reference shows instead of steps: what the trace itself records about
  // the span the issue blamed.
  const legacyDetailByTraceId = useMemo(() => {
    const legacyDetails = new Map<string, IssueEvidenceDetail>();
    for (const [traceId, reference] of issueReferenceByTraceId) {
      if (reference.evidence || !reference.spanId) {
        continue;
      }
      const trace = fetchedTraceByTraceId.get(traceId);
      const detail = trace ? toIssueEvidenceDetail(trace, { namedSpanIds: [reference.spanId] }) : undefined;
      if (detail) {
        legacyDetails.set(traceId, detail);
      }
    }
    return legacyDetails;
  }, [fetchedTraceByTraceId, issueReferenceByTraceId]);

  // The span the issue credited one trace's evidence to, so the trace never contradicts what its
  // card said; a trace the issue named no span for resolves to none and opens on itself.
  const resolveOffendingSpanId = useCallback(
    (traceId: string) => {
      const reference = issueReferenceByTraceId.get(traceId);
      const spanId = reference?.spanId ?? reference?.evidence?.steps[0]?.spanId;
      return spanId ? toExplorerSpanId(spanId) || undefined : undefined;
    },
    [issueReferenceByTraceId],
  );

  // A query that is disabled -- an issue id that cannot be searched -- reports `isLoading` while
  // it sits idle, which would otherwise render a skeleton that never resolves.
  const isLoadingTraces = traces.isLoading && traces.fetchStatus !== 'idle';

  const viewTrace = (header: IssueEvidenceHeader, spanId?: string) =>
    onViewTrace(
      header.addressableTraceId,
      spanId ? toExplorerSpanId(spanId) || undefined : resolveOffendingSpanId(header.traceId),
    );

  const toggleExpanded = (traceId: string) => {
    // Seed from the implicit default before the first toggle, so collapsing a second card does
    // not silently reopen the leading one.
    const current = expandedTraceIds ?? (headers.length > 0 ? [headers[0].traceId] : []);
    setExpandedTraceIds(current.includes(traceId) ? current.filter((id) => id !== traceId) : [...current, traceId]);
  };

  return (
    <div
      data-testid="experiment-issue-evidence"
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}
    >
      <IssueDetailSectionTitle>
        <FormattedMessage
          defaultMessage="Evidence"
          description="Experiment issues tab > Issue detail > Evidence section heading"
        />
      </IssueDetailSectionTitle>

      {isLoadingTraces ? (
        <TableSkeleton lines={3} />
      ) : traces.error && headers.length === 0 ? (
        <Alert
          componentId="mlflow.experiment-issues.evidence.load-error"
          type="warning"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="The traces behind this issue could not be loaded."
              description="Experiment issues tab > Evidence section > Shown when the affected traces could not be fetched"
            />
          }
          description={traces.error.message}
          action={
            <Button componentId="mlflow.experiment-issues.evidence.retry" onClick={() => traces.refetch()}>
              <FormattedMessage
                defaultMessage="Retry"
                description="Experiment issues tab > Evidence section > Retries the affected trace search"
              />
            </Button>
          }
        />
      ) : headers.length === 0 ? (
        // An Issue can name no traces: one filed by hand, or one whose traces have since been
        // deleted. Distinct from a failed read, which is handled above.
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No affected traces are recorded for this issue."
            description="Experiment issues tab > Evidence section > Shown when the issue names no affected traces"
          />
        </Typography.Text>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {headers.map((header, index) => (
            <IssueEvidenceCard
              key={header.traceId}
              header={header}
              issueReference={issueReferenceByTraceId.get(header.traceId)}
              spans={spansByTraceId.get(header.traceId)}
              legacyDetail={legacyDetailByTraceId.get(header.traceId)}
              legacySpanId={issueReferenceByTraceId.get(header.traceId)?.spanId}
              isLoadingDetail={
                needsSpans(header.traceId) &&
                !fetchedTraceByTraceId.has(header.traceId) &&
                (details.isLoading || details.isFetching)
              }
              onViewTrace={(spanId) => viewTrace(header, spanId)}
              isExpanded={isExpanded(header.traceId, index)}
              onToggleExpanded={() => toggleExpanded(header.traceId)}
            />
          ))}
          {traces.error && headers.length > 0 && (
            <Alert
              componentId="mlflow.experiment-issues.evidence.page-error"
              type="warning"
              closable={false}
              message={
                <FormattedMessage
                  defaultMessage="More affected traces could not be loaded."
                  description="Experiment issues tab > Evidence section > Shown when a later page of traces fails"
                />
              }
              description={traces.error.message}
            />
          )}
          {traces.hasNextPage && (
            <div css={{ display: 'flex', justifyContent: 'center' }}>
              <Button
                componentId="mlflow.experiment-issues.evidence.show-more"
                type="link"
                size="small"
                loading={traces.isFetchingNextPage}
                onClick={() => traces.fetchNextPage()}
              >
                <FormattedMessage
                  defaultMessage="Show {count} more"
                  description="Experiment issues tab > Evidence section > Loads the next page of evidence cards"
                  values={{ count: ISSUE_EVIDENCE_PAGE_SIZE }}
                />
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Outside the branches above on purpose: a deep-linked trace must still open when no card
          could be loaded. */}
      <IssueTraceDrawer
        experimentId={experimentId}
        traceInfos={traces.traceInfos}
        getOffendingSpanId={resolveOffendingSpanId}
        onSelectTrace={onViewTrace}
      />
    </div>
  );
};
