import { isArray, isObject } from 'lodash';
import { useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';

import { Alert, Button, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import {
  TagAssignmentKey,
  TagAssignmentLabel,
  TagAssignmentRemoveButton,
  TagAssignmentRoot,
  TagAssignmentRow,
  TagAssignmentValue,
  useTagAssignmentForm,
} from '@databricks/web-shared/unified-tagging';

import { useUpdateTraceTagsMutation } from './useUpdateTraceTagsMutation';
import type { ModelTrace } from '../ModelTrace.types';
import { MLFLOW_INTERNAL_PREFIX } from '../TagUtils';

const emptyValue = { key: '', value: '' };

type KeyValueEntity = {
  key: string;
  value: string;
};

export const useUnifiedTraceTagsModal = ({
  componentIdPrefix,
  onSuccess,
  onClose,
}: {
  componentIdPrefix: string;
  onSuccess?: () => void;
  onClose?: () => void;
}) => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const baseComponentId = `${componentIdPrefix}.tag-assignment-modal`;
  const intl = useIntl();

  const [editedTraceInfo, setEditedTraceInfo] = useState<ModelTrace['info'] | undefined>(undefined);

  const currentTags = useMemo(() => {
    if (!editedTraceInfo) {
      return [];
    }
    if (isArray(editedTraceInfo.tags)) {
      return editedTraceInfo.tags.filter(({ key }) => !key.startsWith(MLFLOW_INTERNAL_PREFIX));
    }
    if (isObject(editedTraceInfo.tags)) {
      return Object.entries(editedTraceInfo.tags)
        .map(([key, value]) => ({ key, value: String(value) }))
        .filter(({ key }) => !key.startsWith(MLFLOW_INTERNAL_PREFIX));
    }
    return [];
  }, [editedTraceInfo]);

  const [isVisible, setIsVisible] = useState(false);
  const { theme } = useDesignSystemTheme();
  const form = useForm<{ tags: KeyValueEntity[] }>({ mode: 'onChange' });

  const tagsForm = useTagAssignmentForm({
    name: 'tags',
    emptyValue: { key: '', value: '' },
    keyProperty: 'key',
    valueProperty: 'value',
    form,
    defaultValues: currentTags,
  });

  const showTagAssignmentModal = (traceInfo: ModelTrace['info']) => {
    setEditedTraceInfo(traceInfo);

    setIsVisible(true);
  };

  const hideTagAssignmentModal = () => {
    setIsVisible(false);
    setEditedTraceInfo(undefined);
    form.reset({ tags: [emptyValue] });
    onClose?.();
  };

  const {
    mutate: commitUpdatedTags,
    isLoading,
    error,
  } = useUpdateTraceTagsMutation({
    onSuccess: () => {
      hideTagAssignmentModal();
      onSuccess?.();
    },
  });

  const handleSubmit = async ({ tags: updatedTags }: { tags: KeyValueEntity[] }) => {
    if (!editedTraceInfo) {
      return;
    }
    const tags: KeyValueEntity[] = updatedTags.filter(({ key }) => Boolean(key));
    const newTags = tags.filter((tag) => !currentTags?.some((t) => t.key === tag.key && t.value === tag.value)) ?? [];
    const deletedTags = currentTags?.filter((tag) => !tags.some((t) => t.key === tag.key)) ?? [];

    // prettier-ignore
    commitUpdatedTags({
      newTags,
      deletedTags,
      modelTraceInfo: editedTraceInfo,
    });
  };

  const TagAssignmentModal = (
    <Modal
      componentId="codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_121"
      title={<FormattedMessage defaultMessage="Add tags" description="Title for unified trace tag assignment modal" />}
      visible={isVisible}
      destroyOnClose
      onCancel={hideTagAssignmentModal}
      footer={
        <>
          <Button
            componentId="codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_130"
            onClick={hideTagAssignmentModal}
            disabled={isLoading}
          >
            <FormattedMessage
              defaultMessage="Cancel"
              description="Cancel button in unified trace tag assignment modal"
            />
          </Button>
          <Button
            componentId="codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_141"
            type="primary"
            onClick={form.handleSubmit(handleSubmit)}
            loading={isLoading}
            disabled={isLoading}
          >
            <FormattedMessage defaultMessage="Save" description="Save button in unified trace tag assignment modal" />
          </Button>
        </>
      }
    >
      {error && (
        <Alert
          type="error"
          message={error instanceof Error ? error.message : String(error)}
          componentId="codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_157"
          closable={false}
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}
      <TagAssignmentRoot {...tagsForm}>
        <TagAssignmentRow>
          <TagAssignmentLabel>
            <FormattedMessage defaultMessage="Key" description="Tag key label in unified trace tag assignment modal" />
          </TagAssignmentLabel>
          <TagAssignmentLabel>
            <FormattedMessage
              defaultMessage="Value"
              description="Tag value label in unified trace tag assignment modal"
            />
          </TagAssignmentLabel>
        </TagAssignmentRow>

        {tagsForm.fields.map((field, index) => {
          return (
            <TagAssignmentRow key={field.id}>
              <TagAssignmentKey
                index={index}
                rules={{
                  validate: {
                    unique: (value) => {
                      const tags = tagsForm.getTagsValues();
                      if (tags?.findIndex((tag) => tag.key === value) !== index) {
                        return intl.formatMessage({
                          defaultMessage: 'Key must be unique',
                          description: 'Error message for unique key in trace tag assignment modal',
                        });
                      }
                      return true;
                    },
                    required: (value) => {
                      const tags = tagsForm.getTagsValues();
                      if (tags?.at(index)?.value && !value) {
                        return intl.formatMessage({
                          defaultMessage: 'Key is required if value is present',
                          description: 'Error message for required key in trace tag assignment modal',
                        });
                      }
                      return true;
                    },
                  },
                }}
              />
              <TagAssignmentValue index={index} />
              <TagAssignmentRemoveButton
                index={index}
                componentId="codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_207"
              />
            </TagAssignmentRow>
          );
        })}
      </TagAssignmentRoot>
    </Modal>
  );

  return {
    TagAssignmentModal,
    showTagAssignmentModal,
    hideTagAssignmentModal,
  };
};
