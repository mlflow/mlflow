import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { Modal, Button, Alert, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { FieldValues } from 'react-hook-form';
import type { KeyValueEntity } from '../types';
import { UnifiedTaggingForm } from '../components/UnifiedTaggingForm';
import { useMemoDeep } from './useMemoDeep';

interface Params {
  componentIdPrefix: string;
  title?: React.ReactNode;
  isLoading?: boolean;
  visible?: boolean;
  initialTags?: KeyValueEntity[];
  error?: string;
  onSubmit: (newTags: KeyValueEntity[], deletedTags: KeyValueEntity[]) => Promise<any>;
  onSuccess?: () => void;
  onClose?: () => void;
  /** Disables Save until the form is valid and differs from `initialTags`. */
  disableSaveWithoutValidChanges?: boolean;
  /**
   * Catches a rejected `onSubmit` so it isn't reported as an unhandled rejection. The modal stays
   * open on a rejection either way; callers show the failure through `error`.
   */
  catchSubmitError?: boolean;
}

const keyProperty = 'key' as const;
const valueProperty = 'value' as const;
const formName = 'tags';

const emptyValue = { key: '', value: '' };

export const useTagAssignmentModal = ({
  componentIdPrefix,
  title,
  visible,
  initialTags,
  isLoading = false,
  error,
  onSubmit,
  onSuccess,
  onClose,
  disableSaveWithoutValidChanges = false,
  catchSubmitError = false,
}: Params) => {
  const baseComponentId = `${componentIdPrefix}.tag-assignment-modal`;

  const memoizedInitialTags = useMemoDeep(() => initialTags, [initialTags]);

  const [isVisible, setIsVisible] = useState(false);
  const { theme } = useDesignSystemTheme();
  const form = useForm<{ tags: KeyValueEntity[] }>({ mode: 'onChange' });

  const showTagAssignmentModal = () => {
    setIsVisible(true);
  };

  const hideTagAssignmentModal = () => {
    setIsVisible(false);
    form.reset({ [formName]: [emptyValue] });
    onClose?.();
  };

  const getTagChanges = (formTags: FieldValues[]) => {
    const tags = formTags.filter((tag) => Boolean(tag[keyProperty])) as KeyValueEntity[];

    // Find tags that are new (not in initialTags at all based on key) or updated (key exists but value changed)
    const newTags =
      tags.filter(
        (tag) =>
          !initialTags?.some((t) => t[keyProperty] === tag[keyProperty] && t[valueProperty] === tag[valueProperty]),
      ) ?? [];

    // Find tags that were deleted (in initialTags but not in current tags based on key)
    const deletedTags = initialTags?.filter((tag) => !tags.some((t) => t[keyProperty] === tag[keyProperty])) ?? [];

    return { newTags, deletedTags };
  };

  const hasValidChanges = () => {
    const { newTags, deletedTags } = getTagChanges(form.watch(formName) ?? []);
    return form.formState.isValid && (newTags.length > 0 || deletedTags.length > 0);
  };

  const handleSubmit = (data: FieldValues) => {
    const { newTags, deletedTags } = getTagChanges(data[formName]);

    onSubmit(newTags, deletedTags).then(
      () => {
        hideTagAssignmentModal();
        onSuccess?.();
      },
      catchSubmitError ? () => undefined : undefined,
    );
  };

  const defaultTitleNode = (
    <FormattedMessage defaultMessage="Add tags" description="Tag assignment modal > Title of the add tags modal" />
  );

  const TagAssignmentModal = (
    <Modal
      componentId="codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_82"
      title={title ?? defaultTitleNode}
      visible={visible ?? isVisible}
      destroyOnClose
      onCancel={hideTagAssignmentModal}
      footer={
        <>
          <Button
            componentId="codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_91"
            onClick={hideTagAssignmentModal}
            disabled={isLoading}
          >
            <FormattedMessage defaultMessage="Cancel" description="Tag assignment modal > Cancel button" />
          </Button>
          <Button
            componentId="codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_99"
            type="primary"
            onClick={form.handleSubmit(handleSubmit)}
            loading={isLoading}
            disabled={isLoading || (disableSaveWithoutValidChanges && !hasValidChanges())}
          >
            <FormattedMessage defaultMessage="Save" description="Tag assignment modal > Save button" />
          </Button>
        </>
      }
    >
      {error && (
        <Alert
          type="error"
          message={error}
          componentId="codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_115"
          closable={false}
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}
      <UnifiedTaggingForm name={formName} form={form} initialTags={memoizedInitialTags} />
    </Modal>
  );

  return {
    TagAssignmentModal,
    showTagAssignmentModal,
    hideTagAssignmentModal,
  };
};

export type { Params as TagAssignmentModalParams };
