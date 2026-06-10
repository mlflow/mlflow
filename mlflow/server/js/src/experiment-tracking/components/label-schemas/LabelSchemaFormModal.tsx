import { useEffect, useMemo, useState } from 'react';

import { Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ModelTraceExplorerResizablePane } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage } from 'react-intl';
import { useForm, useWatch } from 'react-hook-form';

import { LabelSchemaFormRenderer } from './LabelSchemaFormRenderer';
import { LabelSchemaPreview } from './LabelSchemaPreview';
import { useCreateLabelSchemaMutation } from './hooks/useCreateLabelSchemaMutation';
import { useUpdateLabelSchemaMutation } from './hooks/useUpdateLabelSchemaMutation';
import {
  DEFAULT_FORM_VALUES,
  buildLabelSchemaInputFromForm,
  getFormValuesFromSchema,
  validateLabelSchemaForm,
  type LabelSchemaFormData,
} from './labelSchemaFormUtils';
import type { LabelSchema } from './types';

const CID = 'mlflow.label-schema-form-modal';

// Height of the two-pane (fields | preview) body. Fixed so the resizable pane
// has a bound to compute against; the panes scroll internally when taller.
const FORM_BODY_HEIGHT = 480;

export interface LabelSchemaFormModalProps {
  experimentId: string;
  /** When non-null, the modal opens in edit mode pre-populated from the schema. */
  editingSchema: LabelSchema | null;
  visible: boolean;
  onClose: () => void;
  /** Called after a new schema is successfully created (not on edit). */
  onCreated?: (schema: LabelSchema) => void;
}

/**
 * Modal for creating or editing a review question (label schema): a two-pane
 * form (fields | live preview) over the label-schema create/update mutations.
 * Surfaced from the Review tab's "Review questions" manager (which hides itself
 * while this is open); the form itself is the schema authoring UI shared with
 * the rest of the app.
 */
export const LabelSchemaFormModal = ({ experimentId, editingSchema, visible, onClose, onCreated }: LabelSchemaFormModalProps) => {
  const { theme } = useDesignSystemTheme();
  const isEdit = editingSchema != null;
  const defaultValues = editingSchema ? getFormValuesFromSchema(editingSchema) : DEFAULT_FORM_VALUES;
  const {
    control,
    handleSubmit,
    reset,
    formState: { isSubmitted },
  } = useForm<LabelSchemaFormData>({ defaultValues });
  // `useWatch({ control })` returns a deep-partial; backfill with the defaults
  // so the preview and validator always see a complete value with `inputKind`.
  const watchedPartial = useWatch({ control });
  const watched = useMemo<LabelSchemaFormData>(() => ({ ...DEFAULT_FORM_VALUES, ...watchedPartial }), [watchedPartial]);

  const [leftPaneWidth, setLeftPaneWidth] = useState(0);

  const createMutation = useCreateLabelSchemaMutation();
  const updateMutation = useUpdateLabelSchemaMutation();
  const isSubmitting = createMutation.isCreating || updateMutation.isUpdating;
  const submitError = createMutation.error ?? updateMutation.error;

  // Reset the form whenever the modal opens or the edited schema changes. The
  // modal stays mounted across open/close, so without this react-hook-form's
  // `isSubmitted`, stale field values, and mutation errors carry over.
  useEffect(() => {
    if (!visible) {
      return;
    }
    reset(defaultValues);
    createMutation.reset();
    updateMutation.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, editingSchema?.schema_id]);

  const onSubmit = async (form: LabelSchemaFormData) => {
    const errors = validateLabelSchemaForm(form);
    if (Object.keys(errors).length > 0) {
      return;
    }
    const input = buildLabelSchemaInputFromForm(form);
    try {
      if (isEdit && editingSchema) {
        await updateMutation.updateLabelSchemaAsync({
          schema_id: editingSchema.schema_id,
          instruction: form.instruction,
          enable_comment: form.enable_comment,
          input,
        });
      } else {
        const { label_schema } = await createMutation.createLabelSchemaAsync({
          experiment_id: experimentId,
          name: form.name,
          type: form.type,
          input,
          instruction: form.instruction === '' ? undefined : form.instruction,
          enable_comment: form.enable_comment,
        });
        onCreated?.(label_schema);
      }
    } catch {
      // Errors surface via `submitError`; keep the modal open.
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
  const visibleErrors = isSubmitted ? validationErrors : {};

  return (
    <Modal
      componentId={`${CID}.modal`}
      visible={visible}
      size="wide"
      title={
        isEdit ? (
          <FormattedMessage defaultMessage="Edit question" description="Edit review question modal title" />
        ) : (
          <FormattedMessage defaultMessage="Add question" description="Add review question modal title" />
        )
      }
      okText={
        isEdit ? (
          <FormattedMessage defaultMessage="Save" description="Save review question button" />
        ) : (
          <FormattedMessage defaultMessage="Create" description="Create review question button" />
        )
      }
      okButtonProps={{ loading: isSubmitting, disabled: Object.keys(validationErrors).length > 0 }}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Review question modal cancel button" />}
      onOk={handleSubmit(onSubmit)}
      onCancel={handleCancel}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, minHeight: 0 }}>
        <div css={{ display: 'flex', height: FORM_BODY_HEIGHT, minHeight: 0, overflow: 'hidden' }}>
          <ModelTraceExplorerResizablePane
            initialRatio={0.55}
            paneWidth={leftPaneWidth}
            setPaneWidth={setLeftPaneWidth}
            leftMinWidth={320}
            rightMinWidth={320}
            leftChild={
              <div
                css={{
                  flex: 1,
                  width: '100%',
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.sm,
                  paddingTop: theme.spacing.sm,
                  paddingBottom: theme.spacing.md,
                  paddingRight: theme.spacing.md,
                  height: '100%',
                  minHeight: 0,
                }}
              >
                <LabelSchemaFormRenderer
                  control={control}
                  isEdit={isEdit}
                  errors={visibleErrors}
                  watchedValues={{ inputKind: watched.inputKind }}
                />
              </div>
            }
            rightChild={
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  height: '100%',
                  width: '100%',
                  minHeight: 0,
                  overflow: 'hidden',
                  paddingLeft: theme.spacing.md,
                  borderLeft: `1px solid ${theme.colors.border}`,
                }}
              >
                <LabelSchemaPreview formData={watched} />
              </div>
            }
          />
        </div>
        {submitError && (
          <Typography.Text color="error">
            <FormattedMessage
              defaultMessage="Failed to save: {message}"
              description="Review question save error"
              values={{ message: submitError.message }}
            />
          </Typography.Text>
        )}
      </div>
    </Modal>
  );
};
