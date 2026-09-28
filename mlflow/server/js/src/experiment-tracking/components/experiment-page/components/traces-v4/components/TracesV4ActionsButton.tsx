import {
  Button,
  ChevronDownIcon,
  DropdownMenu,
  OverflowIcon,
  TrashIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { compact } from 'lodash';
import { useContext, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { convertTraceInfoV3ToRunEvalEntry, GenAITracesTableContext } from '@databricks/web-shared/genai-traces-table';
import { ToolbarCollapsibleLabel } from '@databricks/web-shared/traces-table';
// Not re-exported from the genai-traces-table barrel — import from its module.
import { GenAITraceComparisonModal } from '@databricks/web-shared/genai-traces-table/components/GenAITraceComparisonModal';
import { type TracesV4TraceActions } from '../hooks/useTracesV4TraceActions';

// The comparison drawer supports comparing 2–3 traces at once, matching `GenAITracesTableActions`.
const MIN_COMPARE = 2;
const MAX_COMPARE = 3;

interface TracesV4ActionsButtonProps {
  /** Number of currently-selected traces — shown in the button label. */
  selectionCount: number;
  // Note: the Databricks build can disable Delete for UC-backed traces; OSS traces are always
  // deletable, so this button has no disabled/disabledReason props and Delete is always enabled.
  onDelete: () => void;
  experimentId: string;
  /** Shared trace-action building blocks powering the "Use for evaluation" group. */
  actions: TracesV4TraceActions;
  /** The full cross-page selection — each carries its `ModelTraceInfoV3`. */
  selectedTraceInfos: ModelTraceInfoV3[];
}

/**
 * Bulk-actions dropdown, shown only when traces are selected. Offers a "Use for evaluation" group
 * (Run scorers, Add to evaluation dataset, Flag for review — each feature-gated via `actions`) above
 * v4's own Delete item. The count is surfaced in the trigger so the user always sees how many traces
 * an action will affect.
 *
 * The eval actions operate on `selectedTraceInfos` — the entire cross-page selection — because the
 * selection store now carries each selected trace's `ModelTraceInfoV3`, not just the on-page rows.
 *
 * Layout mirrors `GenAITracesTableActions`' three sections: Compare on top (enabled for a 2–3 trace
 * selection), the "Use for evaluation" group (Run scorers, Add to evaluation dataset, Flag for review),
 * then an "Edit" group (Edit tags for a single selection, Delete).
 *
 * "Flag for review" routes the selection into a review queue via `actions.AddToReviewQueueDropdown`,
 * a controlled popover anchored on the trigger (mirroring `GenAITracesTableActions`): picking it flips
 * `showReviewQueue`, which swaps the menu for the review-queue picker on the same button.
 */
export const TracesV4ActionsButton = ({
  selectionCount,
  onDelete,
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

  // When true, the review-queue picker replaces the menu on the same trigger (v3's pattern).
  const [showReviewQueue, setShowReviewQueue] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const { AddToReviewQueueDropdown } = actions;

  // Run-judges takes bare trace ids; the dataset modal takes eval entries.
  const selectedTraceIds = useMemo(
    () => compact(selectedTraceInfos.map((trace) => trace.trace_id)),
    [selectedTraceInfos],
  );
  const selectedEntries = useMemo(
    () => selectedTraceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace)),
    [selectedTraceInfos],
  );

  // Compare needs 2–3 traces; Edit tags acts on a single trace (both match `GenAITracesTableActions`).
  const canCompare = selectionCount >= MIN_COMPARE && selectionCount <= MAX_COMPARE;
  const canEditTags = selectionCount === 1;

  const trigger = (
    <Button
      componentId="mlflow.traces-v4.actions.trigger"
      // The button only renders while traces are selected, so it's always primary-styled here to
      // signal that bulk actions are available for the current selection.
      type="primary"
      icon={<OverflowIcon />}
      endIcon={<ChevronDownIcon />}
      // Names the button when its label collapses to icon-only.
      aria-label={intl.formatMessage({
        defaultMessage: 'Actions for selected traces',
        description: 'Aria label for the bulk actions dropdown trigger in the V4 traces toolbar',
      })}
    >
      <ToolbarCollapsibleLabel>
        <FormattedMessage
          defaultMessage="Actions ({count})"
          description="Label for the bulk actions dropdown trigger, showing how many traces are selected"
          values={{ count: selectionCount }}
        />
      </ToolbarCollapsibleLabel>
    </Button>
  );

  return showReviewQueue ? (
    // While flagging, the review-queue picker takes over the same trigger button as its popover
    // anchor; closing it returns to the menu. Only one trigger is ever in the DOM at a time.
    <AddToReviewQueueDropdown
      experimentId={experimentId}
      selectedTraceInfos={selectedTraceInfos}
      open={showReviewQueue}
      popoverAlign="end"
      onOpenChange={(open) => {
        if (!open) {
          setShowReviewQueue(false);
        }
      }}
    >
      {trigger}
    </AddToReviewQueueDropdown>
  ) : (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
        <DropdownMenu.Content align="end">
          <DropdownMenu.Item
            componentId="mlflow.traces-v4.actions.compare"
            onClick={() => setShowCompare(true)}
            disabled={!canCompare}
          >
            <FormattedMessage
              defaultMessage="Compare"
              description="Bulk action that opens a side-by-side comparison of the selected traces"
            />
          </DropdownMenu.Item>
          <DropdownMenu.Separator />
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
            <DropdownMenu.Item
              componentId="mlflow.traces-v4.actions.flag-for-review"
              css={groupItemStyles}
              onClick={() => setShowReviewQueue(true)}
            >
              <FormattedMessage
                defaultMessage="Flag for review"
                description="Bulk action that assigns the selected traces to a review queue"
              />
            </DropdownMenu.Item>
          </DropdownMenu.Group>
          <DropdownMenu.Separator />
          <DropdownMenu.Group>
            <DropdownMenu.Label css={groupLabelStyles}>
              <FormattedMessage
                defaultMessage="Edit"
                description="Group label for the trace-editing bulk actions in the V4 traces toolbar"
              />
            </DropdownMenu.Label>
            <DropdownMenu.Item
              componentId="mlflow.traces-v4.actions.edit-tags"
              css={groupItemStyles}
              disabled={!canEditTags}
              onClick={() => {
                const [trace] = selectedTraceInfos;
                if (trace) {
                  actions.editTags.showEditTagsModalForTrace(trace);
                }
              }}
            >
              <FormattedMessage
                defaultMessage="Edit tags"
                description="Bulk action that edits the tags on the single selected trace"
              />
            </DropdownMenu.Item>
            <DropdownMenu.Item componentId="mlflow.traces-v4.actions.delete" css={groupItemStyles} onClick={onDelete}>
              <DropdownMenu.IconWrapper>
                <TrashIcon />
              </DropdownMenu.IconWrapper>
              <FormattedMessage defaultMessage="Delete" description="Bulk action that deletes the selected traces" />
            </DropdownMenu.Item>
          </DropdownMenu.Group>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
      {showCompare && <GenAITraceComparisonModal traceIds={selectedTraceIds} onClose={() => setShowCompare(false)} />}
    </>
  );
};
