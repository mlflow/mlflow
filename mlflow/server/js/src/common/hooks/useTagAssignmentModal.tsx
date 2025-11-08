import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { Modal, Button, Alert, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { FieldValues } from 'react-hook-form';
import type { KeyValueEntity } from '../types';
import { UnifiedTaggingForm } from '../components/UnifiedTaggingForm';

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
}: Params) => {
  const baseComponentId = `${componentIdPrefix}.tag-assignment-modal`;

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

  const handleSubmit = (data: FieldValues) => {
    const tags: KeyValueEntity[] = data[formName].filter((tag: FieldValues) => Boolean(tag[keyProperty]));
    const newTags =
      tags.filter(
        (tag) =>
          !initialTags?.some((t) => t[keyProperty] === tag[keyProperty] && t[valueProperty] === tag[valueProperty]),
      ) ?? [];
    const deletedTags =
      initialTags?.filter(
        (tag) => !tags.some((t) => t[keyProperty] === tag[keyProperty] && t[valueProperty] === tag[valueProperty]),
      ) ?? [];

    onSubmit(newTags, deletedTags).then(() => {
      hideTagAssignmentModal();
      onSuccess?.();
    });
  };

  const defaultTitleNode = (
    <FormattedMessage defaultMessage="Add tags" description="Tag assignment modal > Title of the add tags modal" />
  );

  const TagAssignmentModal = (
    <Modal
      componentId={`${baseComponentId}`}
      title={title ?? defaultTitleNode}
      visible={visible ?? isVisible}
      destroyOnClose
      onCancel={hideTagAssignmentModal}
      footer={
        <>
          <Button
            componentId={`${baseComponentId}.submit-button`}
            onClick={hideTagAssignmentModal}
            disabled={isLoading}
          >
            <FormattedMessage defaultMessage="Cancel" description="Tag assignment modal > Cancel button" />
          </Button>
          <Button
            componentId={`${baseComponentId}.submit-button`}
            type="primary"
            onClick={form.handleSubmit(handleSubmit)}
            loading={isLoading}
            disabled={isLoading}
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
          componentId={`${baseComponentId}.error`}
          closable={false}
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}
      <UnifiedTaggingForm name={formName} form={form} initialTags={initialTags} />
    </Modal>
  );

  return {
    TagAssignmentModal,
    showTagAssignmentModal,
    hideTagAssignmentModal,
  };
};

export type { Params as TagAssignmentModalParams };
