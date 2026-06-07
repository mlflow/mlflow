import type { ReactNode } from 'react';
import { Button, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export type SaveStatus = 'clean' | 'dirty' | 'invalid' | 'empty-inputs' | 'saving' | 'saved' | 'error';

interface DatasetRecordDetailFooterProps {
  status: SaveStatus;
  /** Error text shown when status = 'error'. */
  errorMessage?: string;
  /**
   * Autosave mode (OSS default): render the save-state indicator only — no Cancel/Save
   * buttons, since edits persist automatically. When false (the Databricks/UC explicit-commit
   * path), the Cancel + Save buttons are shown and `onSave`/`onDiscard` drive them.
   */
  autosave?: boolean;
  onSave?: () => void;
  onDiscard?: () => void;
  /** Primary-button label (explicit mode). Defaults to "Save"; create callers pass "Add record". */
  saveLabel?: ReactNode;
}

const StatusText = ({ status, errorMessage }: { status: SaveStatus; errorMessage?: string }) => {
  const { theme } = useDesignSystemTheme();
  switch (status) {
    case 'saving':
      return (
        <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Spinner size="small" />
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Saving…"
              description="Status text shown in the dataset record drawer while a save is in flight"
            />
          </Typography.Text>
        </span>
      );
    case 'saved':
      return (
        <Typography.Text color="success">
          <FormattedMessage
            defaultMessage="All changes saved"
            description="Status text shown in the dataset record drawer after a successful save"
          />
        </Typography.Text>
      );
    case 'dirty':
      return (
        <Typography.Text color="warning">
          <FormattedMessage
            defaultMessage="Unsaved changes"
            description="Status text shown when the dataset record drawer has unsaved edits"
          />
        </Typography.Text>
      );
    case 'invalid':
      // The per-section editors render their own "Invalid JSON" indicator (icon + red border).
      // This footer line tells the user *why* Save is disabled without re-stating which field.
      return (
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="Fix invalid JSON to save"
            description="Status text shown in the dataset record drawer footer when at least one editor has malformed JSON"
          />
        </Typography.Text>
      );
    case 'empty-inputs':
      // Empty inputs would round-trip to the server as `inputs: []` with `update_mask=inputs`,
      // wiping the field. Spell out the constraint so the user knows what to add.
      return (
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="Inputs cannot be empty"
            description="Status text shown in the dataset record drawer footer when the inputs editor is empty (would otherwise wipe the field server-side)"
          />
        </Typography.Text>
      );
    case 'error':
      return (
        <Typography.Text color="error">
          {errorMessage ?? (
            <FormattedMessage
              defaultMessage="Save failed"
              description="Fallback status text shown when a dataset record save fails"
            />
          )}
        </Typography.Text>
      );
    case 'clean':
    default:
      return null;
  }
};

export const DatasetRecordDetailFooter = ({
  status,
  errorMessage,
  autosave = false,
  onSave,
  onDiscard,
  saveLabel,
}: DatasetRecordDetailFooterProps) => {
  const { theme } = useDesignSystemTheme();
  const canSave = status === 'dirty' || status === 'error';
  const canDiscard = status === 'dirty' || status === 'invalid' || status === 'empty-inputs' || status === 'error';

  return (
    <div
      css={{
        // Horizontal padding comes from the side panel's footer wrapper; we only own the
        // top gap that separates the divider from the buttons here.
        position: 'sticky',
        bottom: 0,
        background: theme.colors.backgroundPrimary,
        borderTop: `1px solid ${theme.colors.border}`,
        paddingTop: theme.spacing.md,
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
      }}
    >
      {/*
        Polite live region: save-state transitions (Saving → Saved / Save failed) are
        announced to assistive tech as the user works through the save flow.
      */}
      <div role="status" aria-live="polite" aria-atomic="true">
        <StatusText status={status} errorMessage={errorMessage} />
      </div>
      {!autosave && (
        <>
          <div css={{ flex: 1 }} />
          <Button
            componentId="mlflow.eval-datasets-v2.drawer.discard"
            onClick={onDiscard}
            // `saving` already implies !canDiscard, so we only need the single check.
            disabled={!canDiscard}
          >
            <FormattedMessage
              defaultMessage="Cancel"
              description="Cancel button on the dataset record side-panel footer — discards in-progress changes"
            />
          </Button>
          <Button
            componentId="mlflow.eval-datasets-v2.drawer.save"
            type="primary"
            onClick={onSave}
            disabled={!canSave}
            loading={status === 'saving'}
          >
            {saveLabel ?? (
              <FormattedMessage
                defaultMessage="Save"
                description="Save-changes button on the dataset record drawer footer"
              />
            )}
          </Button>
        </>
      )}
    </div>
  );
};
