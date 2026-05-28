import { useEffect, useMemo, useRef } from 'react';
import { Button, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useForm, useWatch } from 'react-hook-form';

import { useCreateLabelSchemaMutation } from '../../components/label-schemas/hooks/useCreateLabelSchemaMutation';
import { useUpdateLabelSchemaMutation } from '../../components/label-schemas/hooks/useUpdateLabelSchemaMutation';
import type { LabelSchema } from '../../components/label-schemas/types';
import { LabelSchemaFormRenderer } from './LabelSchemaFormRenderer';
import {
  DEFAULT_FORM_VALUES,
  buildLabelSchemaInputFromForm,
  getFormValuesFromSchema,
  validateLabelSchemaForm,
  type LabelSchemaFormData,
} from './labelSchemaFormUtils';

export interface LabelSchemaModalProps {
  experimentId: string;
  /** When non-null, the modal opens in edit mode pre-populated from the schema. */
  editingSchema: LabelSchema | null;
  visible: boolean;
  onClose: () => void;
  /**
   * Optional callback fired whenever the live form data changes. The
   * parent uses this to drive the admin-page preview pane from the
   * modal's in-flight form state so the SME-view preview updates as the
   * author types. Receives `null` when the modal is not visible so the
   * parent can fall back to the selected-schema state.
   */
  onFormDataChange?: (data: LabelSchemaFormData | null) => void;
  /**
   * Fires with the new schema's `schema_id` after a successful create.
   * The parent uses this to point the preview pane at the just-created
   * schema instead of falling back to the previously-selected card.
   */
  onCreateSuccess?: (schemaId: string) => void;
}

export const LabelSchemaModal = ({
  experimentId,
  editingSchema,
  visible,
  onClose,
  onFormDataChange,
  onCreateSuccess,
}: LabelSchemaModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isEdit = editingSchema != null;
  const defaultValues = editingSchema ? getFormValuesFromSchema(editingSchema) : DEFAULT_FORM_VALUES;
  const { control, handleSubmit, reset } = useForm<LabelSchemaFormData>({
    defaultValues,
  });
  // `useWatch({ control })` returns `DeepPartial<LabelSchemaFormData>`;
  // backfill with `DEFAULT_FORM_VALUES` so downstream consumers (the
  // preview pane via `onFormDataChange`, the validator) always see a
  // complete value with `inputKind` defined. Without this, transient
  // partial states (e.g., between mount and the first `reset`) could
  // make `buildLabelSchemaInputFromForm` hit its `never` branch.
  // Memoize so the reference is stable when no form field changed;
  // otherwise downstream useEffect deps would fire on every render.
  const watchedPartial = useWatch({ control });
  const watched = useMemo<LabelSchemaFormData>(() => ({ ...DEFAULT_FORM_VALUES, ...watchedPartial }), [watchedPartial]);

  const createMutation = useCreateLabelSchemaMutation();
  const updateMutation = useUpdateLabelSchemaMutation();
  const isSubmitting = createMutation.isCreating || updateMutation.isUpdating;
  const submitError = createMutation.error ?? updateMutation.error;

  // Track which `editingSchema?.schema_id` the form was last reset to
  // so the pipe effect can skip emitting until reset has propagated.
  // Without this, on open-for-edit the pipe would emit one render of
  // stale `watched` values before the reset effect updates the form,
  // briefly showing pre-open content in the preview pane.
  const lastResetIdRef = useRef<string | null | undefined>(undefined);

  // Pipe live form state up to the admin-page preview pane while the
  // modal is open; clear (null) when the modal closes so the parent
  // falls back to the saved selected-schema preview.
  useEffect(() => {
    if (!onFormDataChange) {
      return;
    }
    // Skip while we're between an editingSchema change and the reset
    // taking effect; the next render after reset will pipe correctly.
    if (lastResetIdRef.current !== (editingSchema?.schema_id ?? null)) {
      return;
    }
    onFormDataChange(visible ? watched : null);
  }, [visible, watched, onFormDataChange, editingSchema?.schema_id]);

  // When the modal switches between create and edit (or between two
  // different schemas in edit mode), reset the form to the new defaults
  // so the controls reflect the latest source-of-truth values rather
  // than the stale mount-time snapshot. The post-success `reset()` in
  // onSubmit handles the create -> create reopen case (schema_id stays
  // undefined, so this effect wouldn't refire).
  useEffect(() => {
    reset(defaultValues);
    lastResetIdRef.current = editingSchema?.schema_id ?? null;
    // Mutation hook state survives modal close/reopen because the
    // parent keeps this component mounted; clear stale error banners
    // when the user opens the modal for a different schema (or pivots
    // from edit -> create) so they don't see an error from a prior
    // attempt against an unrelated schema.
    createMutation.reset();
    updateMutation.reset();
    // We only want to reset when the identity of the source-of-truth
    // changes (create vs. a specific schema), not on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editingSchema?.schema_id]);

  const onSubmit = async (form: LabelSchemaFormData) => {
    const errors = validateLabelSchemaForm(form);
    if (Object.keys(errors).length > 0) {
      // Validation errors are already surfaced inline by the renderer
      // since `errors` is recomputed on every render from the watched
      // values; bail out without calling the server.
      return;
    }
    const input = buildLabelSchemaInputFromForm(form);
    // On edit, the form is the source of truth: whatever the user sees
    // in the modal IS what they want saved, including unchanged-looking
    // fields. We send the full form payload on every save rather than
    // diffing dirty fields, accepting that this can clobber a parallel
    // edit from another tab. The empty-string instruction case is sent
    // verbatim per the server contract ("" replaces the stored value);
    // callers wanting to clear instruction blank the textarea.
    try {
      if (isEdit && editingSchema) {
        await updateMutation.updateLabelSchemaAsync({
          schema_id: editingSchema.schema_id,
          title: form.title,
          instruction: form.instruction,
          enable_comment: form.enable_comment,
          input,
        });
      } else {
        const created = await createMutation.createLabelSchemaAsync({
          experiment_id: experimentId,
          name: form.name,
          type: form.type,
          title: form.title,
          input,
          // On create, omit blank instruction so the server defaults it
          // to None rather than storing "" verbatim.
          instruction: form.instruction === '' ? undefined : form.instruction,
          enable_comment: form.enable_comment,
        });
        // Surface the new schema's id so the parent can point the
        // preview pane at the just-created schema after close.
        onCreateSuccess?.(created.label_schema.schema_id);
      }
    } catch {
      // Errors are surfaced in the UI via `submitError`; keep the modal
      // open so the user sees what went wrong rather than losing the
      // unsaved form to a transient failure.
      return;
    }
    reset(DEFAULT_FORM_VALUES);
    onClose();
  };

  const handleCancel = () => {
    reset(defaultValues);
    onClose();
  };

  const validationErrors = validateLabelSchemaForm(watched);

  return (
    <Modal
      componentId="mlflow.experiment-label-schemas.modal"
      visible={visible}
      title={
        isEdit
          ? intl.formatMessage({
              defaultMessage: 'Edit label schema',
              description: 'Edit label schema modal title',
            })
          : intl.formatMessage({
              defaultMessage: 'New label schema',
              description: 'Create label schema modal title',
            })
      }
      onCancel={handleCancel}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.experiment-label-schemas.modal.cancel" onClick={handleCancel}>
            <FormattedMessage defaultMessage="Cancel" description="Modal cancel button" />
          </Button>
          <Button
            componentId="mlflow.experiment-label-schemas.modal.submit"
            type="primary"
            loading={isSubmitting}
            disabled={Object.keys(validationErrors).length > 0}
            onClick={() => handleSubmit(onSubmit)()}
          >
            {isEdit ? (
              <FormattedMessage defaultMessage="Save" description="Save schema button" />
            ) : (
              <FormattedMessage defaultMessage="Create" description="Create schema button" />
            )}
          </Button>
        </div>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <LabelSchemaFormRenderer
          control={control}
          isEdit={isEdit}
          errors={validationErrors}
          watchedValues={{ type: watched?.type ?? 'feedback', inputKind: watched?.inputKind ?? 'pass_fail' }}
        />
        {submitError && (
          <Typography.Text color="error">
            <FormattedMessage
              defaultMessage="Failed to save: {message}"
              description="Schema save error"
              values={{ message: submitError.message }}
            />
          </Typography.Text>
        )}
      </div>
    </Modal>
  );
};
