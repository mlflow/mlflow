import { FormUI, Input, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { KeyValueEntity } from '../../../../../common/types';
import { useState } from 'react';

export const ExperimentViewRunsControlsActionsAddNewTagModal = ({
  isOpen,
  setIsOpen,
  selectedRunsExistingTagKeys,
  addNewTag,
}: {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  selectedRunsExistingTagKeys: string[];
  addNewTag: (tag: KeyValueEntity) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [tagKey, setTagKey] = useState<string>('');
  const [tagValue, setTagValue] = useState<string>('');

  const isTagKeyAllowedChars = tagKey === '' || /^[^,.:/=\-\s]+$/.test(tagKey);
  const isTagKeyDuplicate = selectedRunsExistingTagKeys.includes(tagKey);
  const isTagKeyValid = isTagKeyAllowedChars && !isTagKeyDuplicate;
  const isTagNonEmptyAndTagKeyValid = tagKey.length > 0 && tagValue.length > 0 && isTagKeyValid;

  const onConfirmTag = () => {
    if (isTagNonEmptyAndTagKeyValid) {
      addNewTag({ key: tagKey, value: tagValue });
      setIsOpen(false);
      setTagKey('');
      setTagValue('');
    }
  };

  return (
    <Modal
      componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsaddnewtagmodal.tsx_34"
      title={<FormattedMessage defaultMessage="Add New Tag" description="Add new key-value tag modal > Modal title" />}
      visible={isOpen}
      onCancel={() => setIsOpen(false)}
      onOk={onConfirmTag}
      okText={<FormattedMessage defaultMessage="Add" description="Add new key-value tag modal > Add button text" />}
      cancelText={
        <FormattedMessage defaultMessage="Cancel" description="Add new key-value tag modal > Cancel button text" />
      }
      okButtonProps={{ disabled: !isTagNonEmptyAndTagKeyValid }}
    >
      <form css={{ display: 'flex', alignItems: 'flex-end', gap: theme.spacing.md }}>
        <div css={{ display: 'flex', gap: theme.spacing.md, flex: 1 }}>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor="key">
              <FormattedMessage defaultMessage="Key" description="Add new key-value tag modal > Key input label" />
            </FormUI.Label>
            <Input
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsaddnewtagmodal.tsx_51"
              value={tagKey}
              onChange={(e) => setTagKey(e.target.value)}
              validationState={isTagKeyValid ? undefined : 'warning'}
              data-testid="add-new-tag-key-input"
            />
            {!isTagKeyAllowedChars && (
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage=", . : / - = and blank spaces are not allowed"
                  description="Add new key-value tag modal > Invalid characters error"
                />
              </FormUI.Hint>
            )}
            {isTagKeyDuplicate && (
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Tag key already exists on one or more of the selected runs. Please choose a different key."
                  description="Add new key-value tag modal > Duplicate tag key error"
                />
              </FormUI.Hint>
            )}
          </div>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor="value">
              <FormattedMessage defaultMessage="Value" description="Add new key-value tag modal > Value input label" />
            </FormUI.Label>
            <Input
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsaddnewtagmodal.tsx_78"
              value={tagValue}
              onChange={(e) => setTagValue(e.target.value)}
              data-testid="add-new-tag-value-input"
            />
          </div>
        </div>
      </form>
    </Modal>
  );
};
