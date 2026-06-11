import { useCallback, useEffect, useRef, useState } from 'react';
import { useIntl } from 'react-intl';
import { useDeleteDatasetMutation, type Dataset } from './useDatasetsQueries';
import type { DatasetNotifyApi } from './useDatasetNotifications';

export interface UseDatasetDeleteOptions {
  experimentId: string;
  notify: DatasetNotifyApi;
  /**
   * Synchronous side effect run immediately after the mutation succeeds, before any polling
   * and before the success toast. The list page uses this to reset its cursor pagination so
   * the post-delete refetch starts at page 1.
   */
  onMutated?: (target: Dataset) => void;
  /**
   * Optional UC-propagation poll. UC-backed deletes return success before subsequent reads
   * reflect the delete, so the list page polls its list-refetch here. The detail page skips
   * polling and navigates away in `onAfterDelete`. The provided signal aborts on unmount.
   *
   * Resolves to `true` when propagation was observed before the cap, `false` when the poll
   * exhausted its attempts without seeing the delete reflected. The mutation itself already
   * succeeded — `false` only signals UC is still catching up, which the success toast then
   * communicates with a "may take a moment to refresh" suffix.
   */
  pollForPropagation?: (target: Dataset, signal: AbortSignal) => Promise<boolean>;
  /** Fires after the mutation (and any polling) succeed. Detail page navigates back here. */
  onAfterDelete?: (target: Dataset) => void;
}

export interface UseDatasetDeleteResult {
  pendingDataset: Dataset | null;
  isDeleting: boolean;
  isPolling: boolean;
  /** Latest mutation error; cleared on the next requestDelete. */
  error: Error | null;
  requestDelete: (dataset: Dataset) => void;
  cancelDelete: () => void;
  confirmDelete: () => Promise<void>;
}

/**
 * V2 evaluation-dataset delete flow: mutation, optional UC-propagation polling, unmount-safe
 * abort, and success/error toast. Shared between the list page (per-row delete) and the detail
 * page (kebab "Delete dataset" action).
 */
export const useDatasetDelete = ({
  experimentId,
  notify,
  onMutated,
  pollForPropagation,
  onAfterDelete,
}: UseDatasetDeleteOptions): UseDatasetDeleteResult => {
  const intl = useIntl();
  const deleteMutation = useDeleteDatasetMutation(experimentId);
  const [pendingDataset, setPendingDataset] = useState<Dataset | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  // Aborted on unmount so the poll closures (`refetch`, `setIsPolling`, `notify`) stop firing
  // on a torn-down component.
  const pollAbortRef = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      pollAbortRef.current?.abort();
    },
    [],
  );

  const requestDelete = useCallback((dataset: Dataset) => {
    setPendingDataset(dataset);
  }, []);

  const cancelDelete = useCallback(() => {
    setPendingDataset(null);
  }, []);

  const confirmDelete = useCallback(async () => {
    if (!pendingDataset) return;
    const target = pendingDataset;
    pollAbortRef.current?.abort();
    const controller = new AbortController();
    pollAbortRef.current = controller;
    try {
      await deleteMutation.mutateAsync(target.dataset_id);
      // Keep the confirmation modal mounted through the propagation poll so the user has
      // visible "Cleaning up…" feedback rather than a vanished modal with a still-flickering
      // row in the background.
      onMutated?.(target);

      // Default to `true` when no poll runs — the detail-page path has no list to converge,
      // so there's nothing for the user to wait on.
      let propagated = true;
      if (pollForPropagation) {
        setIsPolling(true);
        try {
          propagated = await pollForPropagation(target, controller.signal);
        } finally {
          if (!controller.signal.aborted) setIsPolling(false);
        }
      }
      // Aborted = the modal is now owned by a different requestDelete (or the component
      // unmounted). Do not clear `pendingDataset` — that would clobber the new flow's state.
      if (controller.signal.aborted) return;

      const successMessage = propagated
        ? intl.formatMessage(
            {
              defaultMessage: 'Deleted dataset "{name}"',
              description: 'Success toast after deleting a V2 evaluation dataset',
            },
            { name: target.name ?? target.dataset_id },
          )
        : intl.formatMessage(
            {
              defaultMessage: 'Deleted dataset "{name}" (may take a moment to refresh)',
              description:
                'Success toast after deleting a V2 evaluation dataset when UC propagation has not been observed within the polling window',
            },
            { name: target.name ?? target.dataset_id },
          );
      notify.success(successMessage);
      setPendingDataset(null);
      onAfterDelete?.(target);
    } catch (err) {
      if (controller.signal.aborted) return;
      setIsPolling(false);
      // Leave the modal open so the user can retry or cancel; the `error` field surfaces
      // the failure inline.
      notify.error(err);
    }
  }, [pendingDataset, deleteMutation, onMutated, pollForPropagation, onAfterDelete, notify, intl]);

  return {
    pendingDataset,
    isDeleting: deleteMutation.isLoading,
    isPolling,
    error: deleteMutation.error instanceof Error ? deleteMutation.error : null,
    requestDelete,
    cancelDelete,
    confirmDelete,
  };
};
