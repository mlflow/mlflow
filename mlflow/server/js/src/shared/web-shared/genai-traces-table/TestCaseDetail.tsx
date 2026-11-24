/**
 * Test-case detail drawer for a regression-test run.
 *
 * Opens when a row in the regression-test result table is clicked (replacing
 * the generic trace review). Shows the test name + overall Result, the agent
 * Input/Output (rendered with the same conversation renderer as the trace
 * viewer), and a two-column Assertions/Result table: each assertion's text is
 * expandable, and its pass/fail tag carries the same LLM-judge hover card as
 * the traces table. A "Trace" button still jumps to the raw trace.
 */
import {
  Button,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
  ListIcon,
  Spinner,
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
import { EvaluationsReviewAssessmentTag, isAssessmentPassing } from './components/EvaluationsReviewAssessmentTag';
import { getEvaluationResultAssessmentValue } from './components/GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, EvalTraceComparisonEntry, RunEvaluationResultAssessment } from './types';
import { getAjaxUrl, getDefaultHeaders } from './utils/FetchUtils';
import { useNavigate } from './utils/RoutingUtils';
import { readTraceTag, RESULT_ASSESSMENT_NAME } from './utils/TraceUtils';
import { useQuery } from '../query-client/queryClient';
import type { ModelTrace } from '../model-trace-explorer/ModelTrace.types';
import { getExperimentPageTracesTabRoute } from '../model-trace-explorer/routes';
import { SingleChatTurnMessages } from '../model-trace-explorer/session-view/SingleChatTurnMessages';

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
  const navigate = useNavigate();
  const { DrawerComponent } = useContext(GenAITracesTableContext);
  const run = evaluation.currentRunValue;
  const info = run?.traceInfo;
  const traceId = info?.trace_id;

  // mlflow.test.name is already the full pytest nodeid (includes any [case]).
  const testName = readTraceTag(info, 'mlflow.test.name') ?? traceId ?? 'Test case';

  const infoByName = useMemo(() => new Map((assessmentInfos ?? []).map((i) => [i.name, i])), [assessmentInfos]);

  // One row per assertion (a scorer name can repeat, e.g. multiple guidelines).
  // Label with the criterion text (guideline/instructions) when present, else
  // the scorer name. Each row keeps its AssessmentInfo so the Result cell reuses
  // the value tag.
  const byName = run?.responseAssessmentsByName ?? {};
  const assertions = Object.entries(byName)
    .filter(([name]) => name !== RESULT_ASSESSMENT_NAME)
    .flatMap(([name, arr]) =>
      (arr ?? []).flatMap((a: RunEvaluationResultAssessment, i: number) => {
        const assessmentInfo = infoByName.get(name);
        if (!assessmentInfo) return [];
        const guideline = a.metadata?.['guideline'];
        const label =
          typeof guideline === 'string' && guideline.trim() ? guideline : arr.length > 1 ? `${name} ${i + 1}` : name;
        const passed = isAssessmentPassing(assessmentInfo, getEvaluationResultAssessmentValue(a)) === true;
        return [{ label, assessment: a, assessmentInfo, passed }];
      }),
    );
  const allPassed = assertions.length > 0 && assertions.every((a) => a.passed);

  // Fetch the full trace so SingleChatTurnMessages can render the conversation.
  // useQuery survives the IIFE-based remount pattern in GenAiTracesTableBody.
  const {
    data: fullTrace,
    isLoading: isTraceLoading,
    isError: isTraceError,
  } = useQuery<ModelTrace | null>({
    queryKey: ['testCaseDetailTrace', traceId],
    queryFn: async (): Promise<ModelTrace | null> => {
      if (!traceId) return null;
      const headers = getDefaultHeaders(document.cookie);
      const id = encodeURIComponent(traceId);
      const [infoResp, dataResp] = await Promise.all([
        fetch(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${id}`), { headers }).then((r) => (r.ok ? r.json() : null)),
        fetch(getAjaxUrl(`ajax-api/3.0/mlflow/get-trace-artifact?request_id=${id}`), { headers }).then((r) => {
          if (!r.ok) {
            throw new Error(`Failed to fetch trace artifact (status ${r.status})`);
          }
          return r.json();
        }),
      ]);
      return { info: infoResp?.trace?.trace_info ?? {}, data: dataResp } as ModelTrace;
    },
    enabled: Boolean(traceId),
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
        width="40vw"
        title={
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <Button
              componentId="mlflow.regression-test-detail.prev"
              icon={<ChevronLeftIcon />}
              disabled={!onPrev}
              onClick={() => onPrev?.()}
              aria-label={intl.formatMessage({
                defaultMessage: 'Previous test case',
                description: 'Aria label for the previous-test-case button in the regression-test detail drawer',
              })}
            />
            <Button
              componentId="mlflow.regression-test-detail.next"
              icon={<ChevronRightIcon />}
              disabled={!onNext}
              onClick={() => onNext?.()}
              aria-label={intl.formatMessage({
                defaultMessage: 'Next test case',
                description: 'Aria label for the next-test-case button in the regression-test detail drawer',
              })}
            />
            <div css={{ flex: 1, overflow: 'hidden', display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Typography.Title level={4} withoutMargins css={{ fontFamily: 'monospace' }}>
                {testName}
              </Typography.Title>
              {assertions.length > 0 && <ResultPill passed={allPassed} />}
            </div>
          </div>
        }
      >
        <div css={{ display: 'flex', gap: theme.spacing.sm, marginBottom: theme.spacing.lg }}>
          {traceId && experimentId && (
            <Button
              componentId="mlflow.regression-test-detail.open-trace"
              icon={<ListIcon />}
              onClick={() =>
                navigate(`${getExperimentPageTracesTabRoute(experimentId)}?selectedEvaluationId=${traceId}`)
              }
            >
              <FormattedMessage defaultMessage="Trace" description="Button to open the raw trace from the detail" />
            </Button>
          )}
        </div>

        {isTraceLoading ? (
          <div css={{ display: 'flex', justifyContent: 'center', marginBottom: theme.spacing.lg }}>
            <Spinner />
          </div>
        ) : isTraceError ? (
          <div css={{ marginBottom: theme.spacing.lg }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Could not load the trace for this test case."
                description="Error message shown when the regression-test detail drawer fails to load the trace"
              />
            </Typography.Text>
          </div>
        ) : (
          fullTrace && (
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
          )
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
