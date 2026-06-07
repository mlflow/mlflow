import { useEffect } from 'react';
import { Button, CloseIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';
import { LazyJsonRecordEditor } from './LazyJsonRecordEditor';
import { DatasetRecordDetailFooter } from './DatasetRecordDetailFooter';
import { DatasetRecordCollapsibleSection } from './DatasetRecordCollapsibleSection';
import { DatasetRecordDetailHeader } from './DatasetRecordDetailHeader';
import { TagsCell } from './TagsCell';
import { useRecordSaveState } from '../hooks/useRecordSaveState';
// Tiny intl-based date formatter; replaces @databricks/web-shared/date-time which OSS lacks.
const formatDateTime = (
  input: string | undefined,
  intl: { formatDate: (d: Date, opts?: Intl.DateTimeFormatOptions) => string },
) => {
  if (!input) return '';
  return intl.formatDate(new Date(input), {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Renders the metadata Source row value for a given record source. Trace-sourced records
 * render the trace_id as a Typography.Link that opens the trace explorer via the parent's
 * `onOpenTraceModal` callback; human and document sources render as inline text.
 */
const renderSourceValue = (
  source: DatasetRecord['source'],
  onOpenTraceModal: ((traceId: string) => void) | undefined,
) => {
  if (source?.trace) {
    const traceId = source.trace.trace_id;
    return (
      <>
        Trace (
        <Typography.Link
          componentId="mlflow.eval-datasets-v2.side-panel.source-trace-link"
          onClick={() => onOpenTraceModal?.(traceId)}
        >
          {traceId}
        </Typography.Link>
        )
      </>
    );
  }
  if (source?.human) {
    return `Human (${source.human.user_name})`;
  }
  if (source?.document) {
    return `Document (${source.document.doc_uri})`;
  }
  return '-';
};

interface DatasetRecordSidePanelProps {
  datasetId: string;
  /**
   * The record being edited. Normally always present: "+ Add record" optimistically inserts the
   * new record into the list cache before selecting it by id, and once records have loaded the
   * page closes the panel if a selected id isn't found rather than render it empty. It is only
   * `undefined` during the initial records load (e.g. deep-linking straight to a record URL),
   * where the header shows a placeholder until the list resolves.
   */
  record: DatasetRecord | undefined;
  /** Full record set for the dataset — gates save against mixed singleturn/multiturn schemas. */
  existingRecords: DatasetRecord[];
  open: boolean;
  onClose: () => void;
  /** Receives `unknown` so the same handler covers save errors (always Error-shaped
   * after the hook's wrapping) and inline tag-edit errors (the upsert mutation surfaces
   * the raw rejection). */
  onSaveError?: (error: unknown) => void;
  /**
   * Notifies the parent whenever the panel's dirty/guard state flips. With autosave this stays
   * false (edits are flushed on record-switch / close), but the page keeps the hook wired so a
   * future explicit-commit mode could re-arm the discard guard.
   */
  onDirtyChange?: (isDirty: boolean) => void;
  /**
   * Opens the trace explorer modal for a trace-sourced record. The parent owns the modal so it
   * can overlay the entire dataset page (and so the side panel doesn't need to know about SQL
   * warehouse / labeling-schemas plumbing the modal requires).
   */
  onOpenTraceModal?: (traceId: string) => void;
}

/**
 * Inline side panel for editing a dataset record. New records are created up front by the page's
 * "+ Add record" action (an immediate POST) and then opened here in edit mode, so this panel only
 * ever edits an existing record: edits autosave by id (see `useRecordSaveState`), with no separate
 * Save step.
 */
export const DatasetRecordSidePanel = ({
  datasetId,
  record,
  existingRecords,
  open,
  onClose,
  onSaveError,
  onDirtyChange,
  onOpenTraceModal,
}: DatasetRecordSidePanelProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const editFallback = intl.formatMessage({
    defaultMessage: 'Failed to save record',
    description: 'Generic fallback save-error text for the dataset record side panel',
  });

  // Autosave (the OSS default). No per-save success toast — the footer's saved indicator is the
  // feedback, and a toast on every debounced write would be noise. Errors still surface as toasts.
  const editState = useRecordSaveState({
    datasetId,
    record,
    fallbackErrorMessage: editFallback,
    existingRecords,
    onSaveError,
  });

  // With autosave there is nothing to discard: a pending edit is flushed (committed) on
  // record-switch / close by the hook itself, so closing never needs a confirmation prompt.
  // Keeping this false disarms the page-level discard modal + beforeunload guard.
  const shouldGuard = false;

  // Surface dirty state up to the page so it can serialize close / record-switch through a single
  // confirmation modal (currently inert under autosave — shouldGuard is always false).
  useEffect(() => {
    onDirtyChange?.(open && shouldGuard);
  }, [open, shouldGuard, onDirtyChange]);

  // Native browser prompt for tab-close / reload while dirty. Inert while autosave keeps
  // shouldGuard false; retained for a future explicit-commit mode.
  useEffect(() => {
    if (!open || !shouldGuard) return;
    const handler = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      // Modern browsers ignore the string and show a generic prompt, but setting
      // `returnValue` is required to actually trigger the dialog.
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [open, shouldGuard]);

  // Esc anywhere on the page closes the panel. Document-level (vs `<aside>` onKeyDown)
  // because Monaco swallows keydown events before they bubble to ancestor handlers
  // (see the matching note for Cmd+S in `JsonRecordEditor`), and an aside handler would
  // miss editor-focused presses. `defaultPrevented` defers to anything that legitimately
  // wanted Esc first — Monaco's suggest/find widgets, the bulk-delete or trace modals.
  useEffect(() => {
    if (!open) return undefined;
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented) {
        onClose();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open, onClose]);

  const invalidJsonMessage = intl.formatMessage({
    defaultMessage: 'Invalid JSON',
    description:
      'Inline error shown under a dataset record JSON editor when the contents do not parse as a JSON object',
  });

  const closeLabel = intl.formatMessage({
    defaultMessage: 'Close',
    description: 'Aria label for the close button on the V2 dataset record side panel',
  });
  const editPanelLabel = intl.formatMessage({
    defaultMessage: 'Dataset record details',
    description: 'Aria label for the dataset record edit side panel',
  });

  if (!open) return null;

  return (
    <aside
      aria-label={editPanelLabel}
      css={{
        // `flex: 1` makes the aside fill the parent's height so the body's `overflowY: auto`
        // actually scrolls when content is tall. Without it, the aside sizes to content,
        // pushing the footer below the viewport on long records.
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
    >
      {/*
          Outer wrapper owns the horizontal padding so the inner border-bottom
          sits inside the panel content area — matching the body's `lg`
          horizontal padding (the previous `paddingRight: md` on a single
          padded+bordered div made the divider span past the content). Same
          shape as the footer wrapper below.
        */}
      <div
        css={{
          paddingLeft: theme.spacing.lg,
          paddingRight: theme.spacing.lg,
          flexShrink: 0,
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.sm,
            paddingTop: 0,
            paddingBottom: theme.spacing.md,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          {record ? (
            <DatasetRecordDetailHeader recordId={record.dataset_record_id} />
          ) : (
            // Keeps the close button right-aligned while the record query resolves.
            <span />
          )}
          <Button
            componentId="mlflow.eval-datasets-v2.side-panel.close"
            icon={<CloseIcon />}
            aria-label={closeLabel}
            onClick={onClose}
          />
        </div>
      </div>
      <div
        onKeyDown={editState.onContainerKeyDown}
        css={{
          flex: 1,
          overflowY: 'auto',
          paddingLeft: theme.spacing.lg,
          paddingRight: theme.spacing.lg,
          paddingTop: theme.spacing.md,
        }}
      >
        <DatasetRecordCollapsibleSection
          title={
            <FormattedMessage
              defaultMessage="Inputs"
              description="Section title for inputs in the V2 dataset record side panel"
            />
          }
        >
          <LazyJsonRecordEditor
            value={editState.inputs.text}
            onChange={editState.inputs.setText}
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Dataset record inputs',
              description: 'Aria label for the dataset record inputs JSON editor',
            })}
            errorMessage={editState.inputs.isValid ? undefined : invalidJsonMessage}
            onSaveShortcut={editState.save}
          />
        </DatasetRecordCollapsibleSection>

        <DatasetRecordCollapsibleSection
          title={
            <FormattedMessage
              defaultMessage="Expectations"
              description="Section title for expectations in the V2 dataset record side panel"
            />
          }
        >
          <LazyJsonRecordEditor
            value={editState.expectations.text}
            onChange={editState.expectations.setText}
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Dataset record expectations',
              description: 'Aria label for the dataset record expectations JSON editor',
            })}
            errorMessage={editState.expectations.isValid ? undefined : invalidJsonMessage}
            onSaveShortcut={editState.save}
          />
        </DatasetRecordCollapsibleSection>

        <DatasetRecordCollapsibleSection
          title={
            <FormattedMessage
              defaultMessage="Tags"
              description="Section title for the editable tags section in the V2 dataset record side panel"
            />
          }
        >
          {/* Tags persist immediately via upsert. The TagsCell only renders once the saved
              record is in hand — until then the section is empty (matches the rest of this panel,
              where the saved record drives detail rendering). */}
          {record ? <TagsCell record={record} datasetId={datasetId} onSaveError={onSaveError} /> : null}
        </DatasetRecordCollapsibleSection>

        {record && (
          <DatasetRecordCollapsibleSection
            title={
              <FormattedMessage
                defaultMessage="Metadata"
                description="Section title for metadata in the V2 dataset record side panel"
              />
            }
          >
            <dl
              css={{
                display: 'grid',
                gridTemplateColumns: 'auto 1fr',
                gap: `${theme.spacing.xs}px ${theme.spacing.md}px`,
                margin: 0,
              }}
            >
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Created"
                  description="Metadata label for the dataset record creation time"
                />
              </Typography.Text>
              <Typography.Text>{record.create_time ? formatDateTime(record.create_time, intl) : '-'}</Typography.Text>
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Created by"
                  description="Metadata label for the dataset record author"
                />
              </Typography.Text>
              <Typography.Text>{record.created_by ?? '-'}</Typography.Text>
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Last updated"
                  description="Metadata label for the dataset record last-update time"
                />
              </Typography.Text>
              <Typography.Text>
                {record.last_update_time ? formatDateTime(record.last_update_time, intl) : '-'}
              </Typography.Text>
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Source"
                  description="Metadata label for the dataset record source (origin: trace, document, or human)"
                />
              </Typography.Text>
              <Typography.Text>{renderSourceValue(record.source, onOpenTraceModal)}</Typography.Text>
            </dl>
          </DatasetRecordCollapsibleSection>
        )}
      </div>
      <div
        css={{
          // No paddingBottom here: the outer wrappers (PageWrapper +
          // ExperimentPageTabs content wrapper) already contribute ~24px of
          // bottom space on the experiment page, so adding more here makes
          // the gap below the indicator read as visually empty.
          paddingLeft: theme.spacing.lg,
          paddingRight: theme.spacing.lg,
          flexShrink: 0,
        }}
      >
        <DatasetRecordDetailFooter status={editState.status} errorMessage={editState.errorMessage} autosave />
      </div>
    </aside>
  );
};
