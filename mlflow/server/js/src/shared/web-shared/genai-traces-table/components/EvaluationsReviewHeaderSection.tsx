import { isNil } from 'lodash';
import { useState } from 'react';

import { Button, Spacer, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import type { UseQueryResult } from '@databricks/web-shared/query-client';

import { EvaluationTraceDataDrawer } from './EvaluationTraceDataDrawer';
import { VerticalBar } from './VerticalBar';
import { GenAITracesTableActions } from '../GenAITracesTableActions';
import type { RunEvaluationTracesDataEntry } from '../types';
import { prettySizeWithUnit } from '../utils/DisplayUtils';

// Keep in sync with https://src.dev.databricks.com/databricks-eng/universe@679eb50f2399a24f4c7f919ccb55028bd8662316/-/blob/tracing-server/src/dao/TraceEntitySpace.scala?L45
const DROPPED_SPAN_SIZE_TRACE_METADATA_KEY = 'databricks.tracingserver.dropped_spans_size_bytes';

const EvaluationsReviewSingleRunHeaderSection = ({
  experimentId,
  runDisplayName,
  evaluationResult,
  exportToEvalsInstanceEnabled = false,
  traceQueryResult,
  getTrace,
}: {
  experimentId: string;
  runDisplayName?: string;
  evaluationResult: RunEvaluationTracesDataEntry;
  exportToEvalsInstanceEnabled?: boolean;
  traceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
  getTrace?: (traceId?: string) => Promise<ModelTrace | undefined>;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const [selectedTraceDetailsRequestId, setSelectedTraceDetailsRequestId] = useState<string | null>(null);

  const droppedSpanSize = evaluationResult.traceInfo?.trace_metadata?.[DROPPED_SPAN_SIZE_TRACE_METADATA_KEY];
  let prettySizeString: string | undefined;
  if (!isNil(droppedSpanSize)) {
    const fractionDigits = 2;
    const prettySize = prettySizeWithUnit(Number(droppedSpanSize), fractionDigits);
    prettySizeString = `${prettySize.value} ${prettySize.unit}`;
  }

  return (
    <div css={{ width: '100%' }}>
      <div
        css={{
          width: '100%',
          paddingLeft: theme.spacing.md,
          paddingRight: theme.spacing.md,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
        }}
      >
        <Typography.Title level={4}>{runDisplayName}</Typography.Title>
        {evaluationResult.requestId && (
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
            }}
          >
            {exportToEvalsInstanceEnabled && getTrace && (
              <GenAITracesTableActions
                experimentId={experimentId}
                selectedTraces={[evaluationResult]}
                traceInfos={undefined}
              />
            )}
            <Tooltip
              delayDuration={0}
              componentId="mlflow.evaluations_review.see_detailed_trace_view_tooltip"
              content={
                droppedSpanSize
                  ? `The trace spans were not stored due to their large size: ${prettySizeString}`
                  : undefined
              }
              side="left"
            >
              <Button
                componentId="mlflow.evaluations_review.see_detailed_trace_view_button"
                onClick={() => setSelectedTraceDetailsRequestId(evaluationResult.requestId)}
                loading={traceQueryResult.isLoading}
                disabled={!isNil(droppedSpanSize)}
              >
                <FormattedMessage
                  defaultMessage="See detailed trace view"
                  description="Evaluation review > see detailed trace view button"
                />
              </Button>
            </Tooltip>
          </div>
        )}
        {selectedTraceDetailsRequestId &&
          (!isNil(traceQueryResult.data) ? (
            <EvaluationTraceDataDrawer
              onClose={() => {
                setSelectedTraceDetailsRequestId(null);
              }}
              requestId={selectedTraceDetailsRequestId}
              trace={traceQueryResult.data}
            />
          ) : (
            <>
              {intl.formatMessage({
                defaultMessage: 'No trace data available',
                description: 'Evaluation review > no trace data available',
              })}
            </>
          ))}
      </div>
      <Spacer size="md" />
    </div>
  );
};

/**
 * Displays inputs for a given evaluation result, across one or two runs.
 */
export const EvaluationsReviewHeaderSection = ({
  experimentId,
  runDisplayName,
  otherRunDisplayName,
  evaluationResult,
  otherEvaluationResult,
  exportToEvalsInstanceEnabled = false,
  traceQueryResult,
  compareToTraceQueryResult,
}: {
  experimentId: string;
  evaluationResult: RunEvaluationTracesDataEntry;
  runDisplayName?: string;
  otherRunDisplayName?: string;
  otherEvaluationResult?: RunEvaluationTracesDataEntry;
  exportToEvalsInstanceEnabled?: boolean;
  traceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
  compareToTraceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        width: '100%',
        gap: theme.spacing.sm,
      }}
    >
      <EvaluationsReviewSingleRunHeaderSection
        experimentId={experimentId}
        runDisplayName={runDisplayName}
        evaluationResult={evaluationResult}
        exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
        traceQueryResult={traceQueryResult}
      />
      {otherRunDisplayName && otherEvaluationResult && (
        <>
          <VerticalBar />
          <EvaluationsReviewSingleRunHeaderSection
            experimentId={experimentId}
            runDisplayName={otherRunDisplayName}
            evaluationResult={otherEvaluationResult}
            exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
            traceQueryResult={compareToTraceQueryResult}
          />
        </>
      )}
    </div>
  );
};
