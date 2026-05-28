import { useEffect, useMemo, useState } from 'react';
import { Button, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useForm, useWatch } from 'react-hook-form';

import { ModelTraceExplorerResizablePane } from '@databricks/web-shared/model-trace-explorer';

import { useCreateLabelSchemaMutation } from '../../components/label-schemas/hooks/useCreateLabelSchemaMutation';
import { useUpdateLabelSchemaMutation } from '../../components/label-schemas/hooks/useUpdateLabelSchemaMutation';
import type { LabelSchema } from '../../components/label-schemas/types';
import { LabelSchemaFormRenderer } from './LabelSchemaFormRenderer';
import { LabelSchemaPreview } from './LabelSchemaPreview';
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
}

export const LabelSchemaModal = ({ experimentId, editingSchema, visible, onClose }: LabelSchemaModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isEdit = editingSchema != null;
  const defaultValues = editingSchema ? getFormValuesFromSchema(editingSchema) : DEFAULT_FORM_VALUES;
  const {
    control,
    handleSubmit,
    reset,
    formState: { isSubmitted },
  } = useForm<LabelSchemaFormData>({
    defaultValues,
  });
  // `useWatch({ control })` returns `DeepPartial<LabelSchemaFormData>`;
  // backfill with `DEFAULT_FORM_VALUES` so the preview pane and the
  // validator always see a complete value with `inputKind` defined.
  // Without this, transient partial states between mount and first
  // `reset` could make `buildLabelSchemaInputFromForm` hit its `never`
  // branch.
  const watchedPartial = useWatch({ control });
  const watched = useMemo<LabelSchemaFormData>(() => ({ ...DEFAULT_FORM_VALUES, ...watchedPartial }), [watchedPartial]);

  const [leftPaneWidth, setLeftPaneWidth] = useState(0);

  const createMutation = useCreateLabelSchemaMutation();
  const updateMutation = useUpdateLabelSchemaMutation();
  const isSubmitting = createMutation.isCreating || updateMutation.isUpdating;
  const submitError = createMutation.error ?? updateMutation.error;

  // When the modal switches between create and edit (or between two
  // different schemas in edit mode), reset the form to the new defaults
  // so the controls reflect the latest source-of-truth values rather
  // than the stale mount-time snapshot. Mutation hook state survives
  // modal close/reopen because the parent keeps this component mounted;
  // clear stale error banners so users don't see an error from a prior
  // attempt against an unrelated schema.
  useEffect(() => {
    reset(defaultValues);
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
        await createMutation.createLabelSchemaAsync({
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
  // Don't surface inline field errors until the user has attempted at
  // least one submit (judges flow does the same). The submit button
  // itself still gates on the full `validationErrors` regardless.
  const visibleErrors = isSubmitted ? validationErrors : {};

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
      footer={null}
      destroyOnClose
      // Label schemas have no narrow-variant analog to the judges flow's
      // custom-code scorer; the modal is always two-pane, so the wide
      // sizing props apply unconditionally.
      size="wide"
      verticalSizing="maxed_out"
      css={{ width: '100% !important' }}
      dangerouslySetAntdProps={{
        bodyStyle: {
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          overflow: 'hidden',
        },
      }}
    >
      <form
        onSubmit={handleSubmit(onSubmit)}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        <div css={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
          <ModelTraceExplorerResizablePane
            initialRatio={0.55}
            paneWidth={leftPaneWidth}
            setPaneWidth={setLeftPaneWidth}
            leftMinWidth={320}
            rightMinWidth={320}
            leftChild={
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  height: '100%',
                  width: '100%',
                  overflow: 'hidden',
                  minHeight: 0,
                }}
              >
                <div
                  css={{
                    flex: 1,
                    overflowY: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                    paddingTop: theme.spacing.sm,
                    paddingBottom: theme.spacing.md,
                    paddingRight: theme.spacing.md,
                  }}
                >
                  <LabelSchemaFormRenderer
                    control={control}
                    isEdit={isEdit}
                    errors={visibleErrors}
                    watchedValues={{ type: watched.type, inputKind: watched.inputKind }}
                  />
                </div>
              </div>
            }
            rightChild={
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  height: '100%',
                  width: '100%',
                  overflow: 'hidden',
                  minHeight: 0,
                  paddingLeft: theme.spacing.md,
                  borderLeft: `1px solid ${theme.colors.border}`,
                }}
              >
                <LabelSchemaPreview formData={watched} />
              </div>
            }
          />
        </div>
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            gap: theme.spacing.sm,
            paddingTop: theme.spacing.md,
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          {submitError && (
            <Typography.Text color="error" css={{ flex: 1 }}>
              <FormattedMessage
                defaultMessage="Failed to save: {message}"
                description="Schema save error"
                values={{ message: submitError.message }}
              />
            </Typography.Text>
          )}
          <Button componentId="mlflow.experiment-label-schemas.modal.cancel" onClick={handleCancel}>
            <FormattedMessage defaultMessage="Cancel" description="Modal cancel button" />
          </Button>
          <Button
            componentId="mlflow.experiment-label-schemas.modal.submit"
            type="primary"
            htmlType="submit"
            loading={isSubmitting}
            disabled={Object.keys(validationErrors).length > 0}
          >
            {isEdit ? (
              <FormattedMessage defaultMessage="Save" description="Save schema button" />
            ) : (
              <FormattedMessage defaultMessage="Create" description="Create schema button" />
            )}
          </Button>
        </div>
      </form>
    </Modal>
  );
};
