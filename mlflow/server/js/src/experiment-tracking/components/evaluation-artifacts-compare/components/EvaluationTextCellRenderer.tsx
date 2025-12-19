import { TableSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ICellRendererParams } from '@ag-grid-community/core';
import { FormattedMessage } from 'react-intl';
import React from 'react';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';
import { useSelector } from 'react-redux';
import type { ReduxState } from '../../../../redux-types';
import { EvaluationCellEvaluateButton } from './EvaluationCellEvaluateButton';
import { shouldEnablePromptLab } from '../../../../common/utils/FeatureUtils';
import type { UseEvaluationArtifactTableDataResult } from '../hooks/useEvaluationArtifactTableData';
import { JsonPreview } from '../../../../common/components/JsonFormatting';

// Truncate the text in the cell, it doesn't make sense to populate
// more data into the DOM since cells have hidden overflow anyway
const MAX_TEXT_LENGTH = 512;

interface EvaluationTextCellRendererProps extends ICellRendererParams {
  value: string;
  isGroupByColumn?: boolean;
  context: { highlightedText: string };

  data: UseEvaluationArtifactTableDataResult extends (infer U)[] ? U : UseEvaluationArtifactTableDataResult;

  // Valid only for run columns
  run?: RunRowType;
}

/**
 * Internal use component - breaks down the rendered text into chunks and highlights
 * particular part found by the provided substring.
 */
const HighlightedText = React.memo(({ text, highlight }: { text: string; highlight: string }) => {
  const { theme } = useDesignSystemTheme();
  if (!highlight) {
    return <>{text}</>;
  }

  const parts = text.split(new RegExp(`(${highlight})`, 'gi'));

  return (
    <>
      {parts.map((part, i) => (
        <React.Fragment key={i}>
          {part.toLowerCase() === highlight.toLowerCase() ? (
            <span css={{ backgroundColor: theme.colors.yellow200 }}>{part}</span>
          ) : (
            part
          )}
        </React.Fragment>
      ))}
    </>
  );
});

/**
 * Component used to render a single text cell in the evaluation artifacts comparison table.
 */
/* eslint-disable complexity */
export const EvaluationTextCellRenderer = ({
  value,
  context,
  isGroupByColumn,
  run,
  data,
}: EvaluationTextCellRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const { pendingDataLoading, canEvaluateInRunColumn } = usePromptEngineeringContext();
  const isGatewayRoutesLoading = useSelector(
    ({ modelGateway: { modelGatewayRoutesLoading, modelGatewayRoutesLoadingLegacy } }: ReduxState) => {
      return modelGatewayRoutesLoading.loading;
    },
  );

  const isCellEvaluating = run && pendingDataLoading[run.runUuid]?.[data?.key];
  const outputMetadata = (run && data.outputMetadataByRunUuid?.[run.runUuid]) || null;

  const backgroundColor =
    outputMetadata?.isPending || data.isPendingInputRow
      ? theme.colors.backgroundSecondary
      : theme.colors.backgroundPrimary;

  const structuredJSONValue = React.useMemo(() => {
    try {
      const objectData = JSON.parse(value);
      return objectData;
    } catch (e) {
      return null;
    }
  }, [value]);

  return (
    <div
      css={{
        height: '100%',
        whiteSpace: 'normal',
        padding: theme.spacing.sm,
        overflow: 'hidden',
        position: 'relative',
        cursor: 'pointer',
        backgroundColor,
        '&:hover': {
          backgroundColor: theme.colors.actionDefaultBackgroundHover,
        },
      }}
    >
      {isCellEvaluating ? (
        <TableSkeleton lines={3} />
      ) : (
        <>
          {!value ? (
            <Typography.Text color="info" css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
              <FormattedMessage
                defaultMessage="(empty)"
                description="Experiment page > artifact compare view > results table > no result (empty cell)"
              />
            </Typography.Text>
          ) : structuredJSONValue ? (
            <JsonPreview json={JSON.stringify(structuredJSONValue, null, 2)} />
          ) : (
            <span
              css={{
                display: '-webkit-box',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                WebkitBoxOrient: 'vertical',
                WebkitLineClamp: '7',
              }}
            >
              {isGroupByColumn && context.highlightedText ? (
                <HighlightedText text={value} highlight={context.highlightedText} />
              ) : typeof value === 'string' ? (
                value.substring(0, MAX_TEXT_LENGTH)
              ) : (
                /**
                 * There is a transient state when this value is an object used
                 * for EvaluationImageCellRenderer. This will prevent displaying
                 * [object Object] in the cell and cause AgGrid errors.
                 */
                typeof value !== 'object' && value
              )}
            </span>
          )}
        </>
      )}
      {shouldEnablePromptLab() && run && canEvaluateInRunColumn(run) && (
        <div
          css={{
            position: 'absolute',
            left: 8,
            bottom: 8,
            right: 8,
            display: 'flex',
            gap: theme.spacing.sm,
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            {!value && (
              <EvaluationCellEvaluateButton
                disabled={isCellEvaluating}
                isLoading={isGatewayRoutesLoading}
                run={run}
                rowKey={data.key}
              />
            )}
            {(outputMetadata?.isPending || data.isPendingInputRow) && (
              <Typography.Hint size="sm" css={{ fontStyle: 'italic' }}>
                <FormattedMessage
                  defaultMessage="Unsaved"
                  description="Experiment page > artifact compare view > results table > unsaved indicator"
                />
              </Typography.Hint>
            )}
          </div>
          {outputMetadata && !isCellEvaluating && (
            <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
              {outputMetadata.evaluationTime && (
                <Typography.Hint size="sm">
                  {Math.round(outputMetadata.evaluationTime)} ms
                  {outputMetadata.totalTokens ? ',' : ''}
                </Typography.Hint>
              )}
              {outputMetadata.totalTokens && (
                <Typography.Hint size="sm">
                  <FormattedMessage
                    defaultMessage="{totalTokens} total tokens"
                    description="Experiment page > artifact compare view > results table > total number of evaluated tokens"
                    values={{ totalTokens: outputMetadata.totalTokens }}
                  />
                </Typography.Hint>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
