/**
 * Test-case detail drawer for a regression-test run.
 *
 * Opens when a row in the Test cases table is clicked (replacing the generic
 * trace review). Shows the test name + overall Result, the agent Input/Output
 * (rendered with the same conversation renderer as the trace viewer), and a
 * two-column Assertions/Result table: each assertion's text is expandable, and
 * its pass/fail tag carries the same LLM-judge hover card as the traces table.
 * A "Trace" button still jumps to the raw trace.
 */
import {
  Button,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
  ListIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useContext, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import { GenAITracesTableContext } from './GenAITracesTableContext';

import { EvaluationsReviewAssessmentTag } from './components/EvaluationsReviewAssessmentTag';
import { getEvaluationResultAssessmentValue } from './components/GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, EvalTraceComparisonEntry, RunEvaluationResultAssessment } from './types';
import { useQuery } from '../query-client/queryClient';
import type { ModelTrace } from '../model-trace-explorer/ModelTrace.types';
import { SingleChatTurnMessages } from '../model-trace-explorer/session-view/SingleChatTurnMessages';

const isPass = (v: unknown): boolean =>
  typeof v === 'boolean'
    ? v
    : typeof v === 'number'
      ? v >= 0.5
      : typeof v === 'string'
        ? ['yes', 'pass', 'true'].includes(v.trim().toLowerCase())
        : false;

const readTag = (info: any, key: string): string | undefined => {
  const tags = info?.tags;
  if (Array.isArray(tags)) return tags.find((t: any) => t?.key === key)?.value;
  if (tags && typeof tags === 'object' && tags[key] != null) return String(tags[key]);
  const meta = info?.trace_metadata?.[key];
  return meta != null ? String(meta) : undefined;
};

const stringify = (v: any): string => {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
};

// When the table doesn't carry an AssessmentInfo for a scorer (rare), build a
// minimal one so the value tag + hover still render with the right value type.
const fallbackAssessmentInfo = (name: string, a: RunEvaluationResultAssessment): AssessmentInfo => ({
  name,
  displayName: name,
  isKnown: false,
  isOverall: false,
  metricName: name,
  isCustomMetric: false,
  isEditable: false,
  isRetrievalAssessment: false,
  dtype: typeof a.booleanValue === 'boolean' ? 'boolean' : a.numericValue != null ? 'numeric' : 'pass-fail',
  uniqueValues: new Set(),
  docsLink: '',
  missingTooltip: '',
  description: '',
});

const ResultPill = ({ passed }: { passed: boolean }) => (
  <Tag
    componentId="mlflow.regression-test-detail.result-pill"
    color={passed ? 'turquoise' : 'coral'}
    css={{ margin: 0 }}
  >
    {passed ? (
      <FormattedMessage defaultMessage="Passed" description="Pass status pill in the regression-test detail" />
    ) : (
      <FormattedMessage defaultMessage="Failed" description="Fail status pill in the regression-test detail" />
    )}
  </Tag>
);

// Assertion text that clamps to two lines, with a chevron toggle to expand the
// full text so long guideline rubrics don't blow out the row height.
const ExpandableAssertionText = ({ text }: { text: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expanded, setExpanded] = useState(false);
  // Roughly two lines at the column width; longer text gets a chevron toggle.
  const isLong = text.length > 60;
  return (
    <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.xs, minWidth: 0 }}>
      <Typography.Text
        css={
          expanded
            ? { whiteSpace: 'normal', wordBreak: 'break-word' }
            : {
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
                whiteSpace: 'normal',
                wordBreak: 'break-word',
              }
        }
      >
        {text}
      </Typography.Text>
      {isLong && (
        <Button
          componentId="mlflow.regression-test-detail.assertion-expand"
          type="tertiary"
          size="small"
          icon={expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
          aria-label={
            expanded
              ? intl.formatMessage({
                  defaultMessage: 'Collapse assertion text',
                  description: 'Aria label for the chevron that collapses a long assertion text',
                })
              : intl.formatMessage({
                  defaultMessage: 'Expand assertion text',
                  description: 'Aria label for the chevron that expands a long assertion text',
                })
          }
          css={{ flexShrink: 0 }}
          onClick={() => setExpanded((e) => !e)}
        />
      )}
    </div>
  );
};

export const TestCaseDetail = ({
  evaluation,
  experimentId,
  assessmentInfos,
  onClose,
  onPrev,
  onNext,
}: {
  evaluation: EvalTraceComparisonEntry;
  experimentId?: string;
  assessmentInfos?: AssessmentInfo[];
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { DrawerComponent } = useContext(GenAITracesTableContext);
  const run = evaluation.currentRunValue;
  const info: any = run?.traceInfo;
  const traceId: string | undefined = info?.trace_id;

  const baseName = readTag(info, 'mlflow.test.name') ?? traceId ?? 'Test case';
  const caseId = readTag(info, 'mlflow.test.case_id');
  const testName = caseId ? `${baseName}[${caseId}]` : baseName;

  const infoByName = useMemo(() => new Map((assessmentInfos ?? []).map((i) => [i.name, i])), [assessmentInfos]);

  // Flatten every assertion (one row each) -- multiple guideline assertions can
  // share the default "guidelines" name, so we don't collapse to the first.
  // Each row keeps the raw assessment + its AssessmentInfo so the Result cell
  // can render the same value tag + LLM-judge hover the traces table uses.
  const byName = run?.responseAssessmentsByName ?? {};
  const assertions = Object.entries(byName)
    .filter(([name]) => name !== 'Result')
    .flatMap(([name, arr]) =>
      (arr ?? []).map((a: RunEvaluationResultAssessment, i: number) => {
        const value = getEvaluationResultAssessmentValue(a);
        const named = arr.length > 1 ? `${name} ${i + 1}` : name;
        const metaGuideline = stringify(a?.metadata?.['guideline'] ?? a?.metadata?.['guidelines']);
        const label = metaGuideline || named || 'Assertion';
        const assessmentInfo = infoByName.get(name) ?? fallbackAssessmentInfo(name, a);
        return { label, assessment: a, assessmentInfo, passed: isPass(value) };
      }),
    );
  const allPassed = assertions.length > 0 && assertions.every((a) => a.passed);

  // Fetch the full trace (info + spans) so SingleChatTurnMessages can render
  // the deduped user/assistant conversation exactly like the session view does.
  // useQuery survives the IIFE-based remount pattern in GenAiTracesTableBody.
  const { data: fullTrace } = useQuery<ModelTrace | null>({
    queryKey: ['testCaseDetailTrace', traceId],
    queryFn: async (): Promise<ModelTrace | null> => {
      const [infoResp, dataResp] = await Promise.all([
        fetch(`/ajax-api/3.0/mlflow/traces/${encodeURIComponent(traceId!)}`).then((r) => (r.ok ? r.json() : null)),
        fetch(`/ajax-api/3.0/mlflow/get-trace-artifact?request_id=${encodeURIComponent(traceId!)}`).then((r) =>
          r.ok ? r.json() : null,
        ),
      ]);
      if (!dataResp) return null;
      return { info: infoResp?.trace?.trace_info ?? {}, data: dataResp } as ModelTrace;
    },
    enabled: !!traceId,
    staleTime: Infinity,
  });

  return (
    <DrawerComponent.Root
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DrawerComponent.Content
        componentId="mlflow.regression-test-detail.drawer"
        width="60vw"
        title={
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <Button
              componentId="mlflow.regression-test-detail.prev"
              disabled={!onPrev}
              onClick={() => onPrev?.()}
            >
              <ChevronLeftIcon />
            </Button>
            <Button
              componentId="mlflow.regression-test-detail.next"
              disabled={!onNext}
              onClick={() => onNext?.()}
            >
              <ChevronRightIcon />
            </Button>
            <div css={{ flex: 1, overflow: 'hidden', display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Typography.Title level={4} withoutMargins css={{ fontFamily: 'monospace' }}>
                {testName}
              </Typography.Title>
              <ResultPill passed={allPassed} />
            </div>
          </div>
        }
      >
        <div css={{ display: 'flex', gap: theme.spacing.sm, marginBottom: theme.spacing.lg }}>
          {traceId && experimentId && (
            <Button
              componentId="mlflow.regression-test-detail.open-trace"
              icon={<ListIcon />}
              onClick={() => {
                window.location.hash = `#/experiments/${experimentId}/traces?selectedTraceId=${traceId}`;
              }}
            >
              <FormattedMessage defaultMessage="Trace" description="Button to open the raw trace from the detail" />
            </Button>
          )}
        </div>

        {fullTrace && (
          <div
            css={{
              marginBottom: theme.spacing.lg,
              // Add visible borders to the chat message bubbles rendered by
              // SingleChatTurnMessages (the component sets borderWidth but
              // not borderStyle/borderColor on the message elements).
              '& > div > div': {
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusMd,
              },
            }}
          >
            <SingleChatTurnMessages trace={fullTrace} />
          </div>
        )}

        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.legacyBorders.borderRadiusMd,
            overflow: 'hidden',
          }}
        >
          <Table>
            <TableRow isHeader>
              <TableHeader componentId="mlflow.regression-test-detail.col-assertion" css={{ flexGrow: 1 }}>
                <FormattedMessage defaultMessage="Assertions" description="Assertions column header" />
              </TableHeader>
              <TableHeader componentId="mlflow.regression-test-detail.col-result" css={{ flexGrow: 1 }}>
                <FormattedMessage defaultMessage="Result" description="Result column header" />
              </TableHeader>
            </TableRow>
            {assertions.length === 0 ? (
              <TableRow>
                <TableCell css={{ flexGrow: 1 }}>
                  <Typography.Text color="secondary">
                    {intl.formatMessage({
                      defaultMessage: 'No assertions recorded for this test.',
                      description: 'Empty assertions state in the regression-test detail',
                    })}
                  </Typography.Text>
                </TableCell>
              </TableRow>
            ) : (
              assertions.map((a, i) => (
                <TableRow key={`${a.label}-${i}`}>
                  <TableCell css={{ flexGrow: 1, alignItems: 'flex-start' }}>
                    <ExpandableAssertionText text={a.label} />
                  </TableCell>
                  <TableCell css={{ flexGrow: 1, alignItems: 'flex-start' }}>
                    <EvaluationsReviewAssessmentTag
                      type="value"
                      assessment={a.assessment}
                      assessmentInfo={a.assessmentInfo}
                      showRationaleInTooltip
                      hideAssessmentName
                      disableJudgeTypeIcon
                    />
                  </TableCell>
                </TableRow>
              ))
            )}
          </Table>
        </div>
      </DrawerComponent.Content>
    </DrawerComponent.Root>
  );
};
