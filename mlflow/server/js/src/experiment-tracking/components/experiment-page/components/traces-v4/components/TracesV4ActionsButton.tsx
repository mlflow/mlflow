import { Button, ChevronDownIcon, DropdownMenu, TrashIcon, useDesignSystemTheme } from '@databricks/design-system';
import { compact } from 'lodash';
import React, { useContext, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { convertTraceInfoV3ToRunEvalEntry, GenAITracesTableContext } from '@databricks/web-shared/genai-traces-table';
import { type TracesV4TraceActions } from '../hooks/useTracesV4TraceActions';

interface TracesV4ActionsButtonProps {
  /** Number of currently-selected traces — shown in the button label. */
  selectionCount: number;
  onDelete: () => void;
  /** Grays out the Delete item (UC-backed traces can't be deleted here). */
  disabled?: boolean;
  /** Tooltip explaining why Delete is disabled — rendered by `DropdownMenu.Item`. */
  disabledReason?: React.ReactNode;
  experimentId: string;
  /** Shared trace-action building blocks powering the "Use for evaluation" group. */
  actions: TracesV4TraceActions;
  /** The full cross-page selection — each carries its `ModelTraceInfoV3`. */
  selectedTraceInfos: ModelTraceInfoV3[];
}

/**
 * Bulk-actions dropdown, shown only when traces are selected. Offers a "Use for evaluation" group
 * (Evaluate, Run judges, Add to evaluation dataset, Add to labeling session, Flag for review — each
 * feature-gated via `actions`) above v4's own Delete item. The count is surfaced in the trigger so
 * the user always sees how many traces an action will affect.
 *
 * The eval actions operate on `selectedTraceInfos` — the entire cross-page selection — because the
 * selection store now carries each selected trace's `ModelTraceInfoV3`, not just the on-page rows.
 */
export const TracesV4ActionsButton = ({
  selectionCount,
  onDelete,
  disabled,
  disabledReason,
  experimentId,
  actions,
  selectedTraceInfos,
}: TracesV4ActionsButtonProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  // Match the shared GenAITracesTable menu: mute the group heading and indent its items so
  // "Use for evaluation" reads as a section label, not a disabled row.
  const groupLabelStyles = { color: theme.colors.textSecondary };
  const groupItemStyles = { paddingLeft: theme.spacing.lg };

  const { showAddToEvaluationDatasetModal } = useContext(GenAITracesTableContext);

  // Run-judges takes bare trace ids; the dataset modal takes eval entries.
  const selectedTraceIds = useMemo(
    () => compact(selectedTraceInfos.map((trace) => trace.trace_id)),
    [selectedTraceInfos],
  );
  const selectedEntries = useMemo(
    () => selectedTraceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace)),
    [selectedTraceInfos],
  );

  const trigger = (
    <Button
      componentId="mlflow.traces-v4.actions.trigger"
      // The button only renders while traces are selected, so it's always primary-styled here to
      // signal that bulk actions are available for the current selection.
      type="primary"
      endIcon={<ChevronDownIcon />}
      aria-label={intl.formatMessage({
        defaultMessage: 'Actions for selected traces',
        description: 'Aria label for the bulk actions dropdown trigger in the V4 traces toolbar',
      })}
    >
      <FormattedMessage
        defaultMessage="Actions ({count})"
        description="Label for the bulk actions dropdown trigger, showing how many traces are selected"
        values={{ count: selectionCount }}
      />
    </Button>
  );

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        <DropdownMenu.Group>
          <DropdownMenu.Label css={groupLabelStyles}>
            <FormattedMessage
              defaultMessage="Use for evaluation"
              description="Group label for the evaluation-related bulk actions in the V4 traces toolbar"
            />
          </DropdownMenu.Label>
          {actions.runJudges && (
            <DropdownMenu.Item
              componentId="mlflow.traces-v4.actions.run-judges"
              css={groupItemStyles}
              onClick={() => actions.runJudges?.showRunJudgesModal(selectedTraceIds)}
            >
              <FormattedMessage
                defaultMessage="Run scorers"
                description="Bulk action that runs scorers on the selected traces"
              />
            </DropdownMenu.Item>
          )}
          <DropdownMenu.Item
            componentId="mlflow.traces-v4.actions.add-to-dataset"
            css={groupItemStyles}
            onClick={() => showAddToEvaluationDatasetModal?.(selectedEntries)}
          >
            <FormattedMessage
              defaultMessage="Add to evaluation dataset"
              description="Bulk action that adds the selected traces to an evaluation dataset"
            />
          </DropdownMenu.Item>
        </DropdownMenu.Group>
        <DropdownMenu.Separator />
        <DropdownMenu.Item
          componentId="mlflow.traces-v4.actions.delete"
          onClick={onDelete}
          disabled={disabled}
          disabledReason={disabledReason}
        >
          <DropdownMenu.IconWrapper>
            <TrashIcon />
          </DropdownMenu.IconWrapper>
          <FormattedMessage defaultMessage="Delete" description="Bulk action that deletes the selected traces" />
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
