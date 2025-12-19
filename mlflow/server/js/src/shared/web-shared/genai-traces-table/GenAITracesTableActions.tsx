import type { RowSelectionState } from '@tanstack/react-table';
import { compact, isNil } from 'lodash';
import { useCallback, useContext, useMemo, useState } from 'react';

import { Button, Tooltip, DropdownMenu, ChevronDownIcon } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { GenAITracesTableContext } from './GenAITracesTableContext';
import { GenAiDeleteTraceModal } from './components/GenAiDeleteTraceModal';
import type { RunEvaluationTracesDataEntry, TraceActions } from './types';
import { shouldEnableTagGrouping } from './utils/FeatureUtils';
import { applyTraceInfoV3ToEvalEntry, getRowIdFromTrace } from './utils/TraceUtils';
import { GenAITraceComparisonModal } from './components/GenAITraceComparisonModal';
import type { ModelTraceInfoV3 } from '../model-trace-explorer';

interface GenAITracesTableActionsProps {
  experimentId: string;
  // @deprecated
  selectedTraces?: RunEvaluationTracesDataEntry[];
  // @deprecated
  setRowSelection?: React.Dispatch<React.SetStateAction<RowSelectionState>>;
  traceActions?: TraceActions;
  traceInfos: ModelTraceInfoV3[] | undefined;
}

export const GenAITracesTableActions = (props: GenAITracesTableActionsProps) => {
  const { traceActions, experimentId, selectedTraces: selectedTracesFromProps, traceInfos, setRowSelection } = props;

  const { table, selectedRowIds } = useContext(GenAITracesTableContext);

  const selectedTracesFromContext: RunEvaluationTracesDataEntry[] | undefined = useMemo(
    () =>
      applyTraceInfoV3ToEvalEntry(
        selectedRowIds
          .map((rowId) => {
            const traceInfo = traceInfos?.find((trace) => getRowIdFromTrace(trace) === rowId);
            if (!traceInfo) {
              return undefined;
            }
            return {
              evaluationId: traceInfo.trace_id,
              requestId: traceInfo.client_request_id || traceInfo.trace_id,
              inputsId: traceInfo.trace_id,
              inputs: {},
              outputs: {},
              targets: {},
              overallAssessments: [],
              responseAssessmentsByName: {},
              metrics: {},
              traceInfo,
            };
          })
          .filter((trace) => !isNil(trace)),
      ),
    [selectedRowIds, traceInfos],
  );

  const selectedTraces: RunEvaluationTracesDataEntry[] = selectedTracesFromProps || selectedTracesFromContext;

  return (
    <TraceActionsDropdown
      experimentId={experimentId}
      selectedTraces={selectedTraces}
      traceActions={traceActions}
      setRowSelection={setRowSelection ?? table?.setRowSelection}
    />
  );
};

interface TraceActionsDropdownProps {
  experimentId: string;
  selectedTraces: RunEvaluationTracesDataEntry[];
  traceActions?: TraceActions;
  setRowSelection: React.Dispatch<React.SetStateAction<RowSelectionState>> | undefined;
}

const TraceActionsDropdown = (props: TraceActionsDropdownProps) => {
  const { experimentId, selectedTraces, traceActions, setRowSelection } = props;
  const intl = useIntl();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showCompareModal, setShowCompareModal] = useState(false);

  const handleEditTags = useCallback(() => {
    if (selectedTraces.length === 1 && selectedTraces[0].traceInfo && traceActions?.editTags) {
      traceActions.editTags.showEditTagsModalForTrace(selectedTraces[0].traceInfo);
    }
  }, [selectedTraces, traceActions]);

  const handleDeleteTraces = useCallback(() => {
    setShowDeleteModal(true);
  }, []);

  const handleOpenCompare = useCallback(() => {
    setShowCompareModal(true);
  }, []);

  const handleCloseCompare = useCallback(() => {
    setShowCompareModal(false);
  }, []);

  const deleteTraces = useCallback(
    async (experimentId: string, traceIds: string[]) => {
      await traceActions?.deleteTracesAction?.deleteTraces?.(experimentId, traceIds);
      setRowSelection?.({});
    },
    [setRowSelection, traceActions],
  );

  const hasExportAction = Boolean(traceActions?.exportToEvals);
  const hasEditTagsAction = shouldEnableTagGrouping() && Boolean(traceActions?.editTags);
  const hasDeleteAction = Boolean(traceActions?.deleteTracesAction);

  const handleExportToDatasets = useCallback(() => {
    traceActions?.exportToEvals?.setShowExportTracesToDatasetsModal?.(true);
  }, [traceActions?.exportToEvals]);
  const showDatasetModal = traceActions?.exportToEvals?.showExportTracesToDatasetsModal;

  const isEditTagsDisabled = selectedTraces.length > 1;
  const noTracesSelected = selectedTraces.length === 0;
  const noActionsAvailable = !hasExportAction && !hasEditTagsAction && !hasDeleteAction;

  const canCompare = selectedTraces.length >= 2 && selectedTraces.length < 4;

  if (noActionsAvailable) {
    return null;
  }

  const ActionButton = (
    <Button
      componentId="mlflow.genai-traces-table.actions-dropdown"
      disabled={noTracesSelected}
      type="primary"
      endIcon={<ChevronDownIcon />}
    >
      {intl.formatMessage(
        {
          defaultMessage: 'Actions{count}',
          description: 'Trace actions dropdown button',
        },
        {
          count: noTracesSelected ? '' : ` (${selectedTraces.length})`,
        },
      )}
    </Button>
  );

  return (
    <>
      <DropdownMenu.Root>
        {noTracesSelected ? (
          <Tooltip
            componentId="mlflow.genai-traces-table.actions-disabled-tooltip"
            content={intl.formatMessage({
              defaultMessage: 'Select one or more traces to add to an evaluation or edit the traces.',
              description: 'Tooltip shown when actions button is disabled due to no trace selection',
            })}
          >
            <div>
              <DropdownMenu.Trigger asChild>{ActionButton}</DropdownMenu.Trigger>
            </div>
          </Tooltip>
        ) : (
          <DropdownMenu.Trigger asChild>{ActionButton}</DropdownMenu.Trigger>
        )}
        <DropdownMenu.Content>
          <DropdownMenu.Item
            componentId="mlflow.genai-traces-table.compare-traces"
            onClick={handleOpenCompare}
            disabled={!canCompare}
          >
            {intl.formatMessage({ defaultMessage: 'Compare', description: 'Compare traces button' })}
          </DropdownMenu.Item>
          <DropdownMenu.Separator />
          {hasExportAction && (
            <>
              <DropdownMenu.Group>
                <DropdownMenu.Label>
                  {intl.formatMessage({
                    defaultMessage: 'Use for evaluation',
                    description: 'Trace actions dropdown group label',
                  })}
                </DropdownMenu.Label>
                <DropdownMenu.Item
                  componentId="mlflow.genai-traces-table.export-to-datasets"
                  onClick={handleExportToDatasets}
                >
                  {intl.formatMessage({
                    defaultMessage: 'Add to evaluation dataset',
                    description: 'Add traces to evaluation dataset action',
                  })}
                </DropdownMenu.Item>
              </DropdownMenu.Group>
            </>
          )}
          {(hasEditTagsAction || hasDeleteAction) && (
            <>
              {hasExportAction && <DropdownMenu.Separator />}
              <DropdownMenu.Group>
                <DropdownMenu.Label>
                  {intl.formatMessage({
                    defaultMessage: 'Edit',
                    description: 'Trace actions dropdown group label',
                  })}
                </DropdownMenu.Label>
                {hasEditTagsAction && (
                  <DropdownMenu.Item
                    componentId="mlflow.genai-traces-table.edit-tags"
                    onClick={handleEditTags}
                    disabled={isEditTagsDisabled}
                  >
                    {intl.formatMessage({
                      defaultMessage: 'Edit tags',
                      description: 'Edit tags action',
                    })}
                  </DropdownMenu.Item>
                )}
                {hasDeleteAction && (
                  <DropdownMenu.Item
                    componentId="mlflow.genai-traces-table.delete-traces"
                    onClick={handleDeleteTraces}
                    disabled={traceActions?.deleteTracesAction?.isDisabled}
                    disabledReason={traceActions?.deleteTracesAction?.disabledReason}
                  >
                    {intl.formatMessage({
                      defaultMessage: 'Delete traces',
                      description: 'Delete traces action',
                    })}
                  </DropdownMenu.Item>
                )}
              </DropdownMenu.Group>
            </>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Root>

      {traceActions?.editTags?.EditTagsModal}

      {
        // prettier-ignore
        showDatasetModal &&
        traceActions?.exportToEvals?.renderExportTracesToDatasetsModal?.({
          selectedTraceInfos: compact(selectedTraces.map((trace) => trace.traceInfo)),
        })
      }

      {showDeleteModal && traceActions?.deleteTracesAction && (
        <GenAiDeleteTraceModal
          experimentIds={[experimentId]}
          visible={showDeleteModal}
          selectedTraces={selectedTraces}
          handleClose={() => setShowDeleteModal(false)}
          deleteTraces={deleteTraces}
        />
      )}

      {showCompareModal && (
        <GenAITraceComparisonModal
          traceIds={compact(selectedTraces.map((trace) => trace.traceInfo?.trace_id))}
          onClose={handleCloseCompare}
        />
      )}
    </>
  );
};
