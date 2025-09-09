import { isEqual, sortBy } from 'lodash';
import { useCallback, useMemo, useRef, useState } from 'react';
import { truncate } from 'lodash';

import {
  Button,
  FormUI,
  Modal,
  PlusIcon,
  Popover,
  RHFControlledComponents,
  RestoreAntDDefaultClsPrefix,
  LegacyTooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Typography } from '@databricks/design-system';
import type { KeyValueEntity } from '../types';
import { FormattedMessage, useIntl } from 'react-intl';
import { useForm } from 'react-hook-form';
import { TagKeySelectDropdown } from '../components/TagSelectDropdown';
import { KeyValueTag } from '../components/KeyValueTag';
import { ErrorWrapper } from '../utils/ErrorWrapper';

function getTagsMap(tags: KeyValueEntity[]) {
  return new Map(tags.map((tag) => [tag.key, tag]));
}

/**
 * Provides methods to initialize and display modal used to add and remove tags from any compatible entity
 */
export const useEditKeyValueTagsModal = <T extends { tags?: KeyValueEntity[] }>({
  onSuccess,
  saveTagsHandler,
  allAvailableTags,
  valueRequired = false,
  title,
}: {
  onSuccess?: () => void;
  saveTagsHandler: (editedEntity: T, existingTags: KeyValueEntity[], newTags: KeyValueEntity[]) => Promise<any>;
  allAvailableTags?: string[];
  valueRequired?: boolean;
  title?: React.ReactNode;
}) => {
  const editedEntityRef = useRef<T>();
  const [errorMessage, setErrorMessage] = useState<string>('');
  const { theme } = useDesignSystemTheme();

  const [initialTags, setInitialTags] = useState<Map<string, KeyValueEntity>>(new Map());
  const [finalTags, setFinalTags] = useState<Map<string, KeyValueEntity>>(new Map());

  const [showModal, setShowModal] = useState(false);

  const form = useForm<KeyValueEntity>({
    defaultValues: {
      key: undefined,
      value: '',
    },
  });

  const hideModal = () => setShowModal(false);

  /**
   * Function used to invoke the modal and start editing tags of the particular model version
   */
  const showEditTagsModal = useCallback(
    (editedEntity: T) => {
      editedEntityRef.current = editedEntity;
      setInitialTags(getTagsMap(editedEntity.tags || []));
      setFinalTags(getTagsMap(editedEntity.tags || []));
      form.reset();

      setShowModal(true);
    },
    [form],
  );

  const saveTags = async () => {
    if (!editedEntityRef.current) {
      return;
    }
    setErrorMessage('');
    setIsLoading(true);
    saveTagsHandler(editedEntityRef.current, Array.from(initialTags.values()), Array.from(finalTags.values()))
      .then(() => {
        hideModal();
        onSuccess?.();
        setIsLoading(false);
      })
      .catch((e: ErrorWrapper | Error) => {
        setIsLoading(false);
        setErrorMessage(e instanceof ErrorWrapper ? e.getUserVisibleError()?.message : e.message);
      });
  };

  const intl = useIntl();
  const formValues = form.watch();

  const [isLoading, setIsLoading] = useState(false);

  const hasNewValues = useMemo(
    () => !isEqual(sortBy(Array.from(initialTags.values()), 'key'), sortBy(Array.from(finalTags.values()), 'key')),
    [initialTags, finalTags],
  );
  const isDirty = formValues.key || formValues.value;
  const showPopoverMessage = hasNewValues && isDirty;

  const onKeyChangeCallback = (key: string | undefined) => {
    const tag = key ? finalTags.get(key) : undefined;
    /**
     * If a tag value exists for provided key, set the value to the existing tag value
     */
    form.setValue('value', tag?.value ?? '');
  };

  const handleTagDelete = ({ key }: KeyValueEntity) => {
    setFinalTags((currentFinalTags) => {
      currentFinalTags.delete(key);
      return new Map(currentFinalTags);
    });
  };

  const onSubmit = () => {
    // Do not accept form if no value provided while it's required
    if (valueRequired && !formValues.value.trim()) {
      return;
    }

    // Add new tag to existing tags leaving only one tag per key value
    const newEntries = new Map(finalTags);
    newEntries.set(formValues.key, formValues);

    setFinalTags(newEntries);
    form.reset();
  };

  const EditTagsModal = (
    <Modal
      componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_135"
      destroyOnClose
      visible={showModal}
      title={
        title ?? (
          <FormattedMessage
            defaultMessage="Add/Edit tags"
            description="Key-value tag editor modal > Title of the update tags modal"
          />
        )
      }
      onCancel={hideModal}
      footer={
        <RestoreAntDDefaultClsPrefix>
          <Button
            componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_147"
            dangerouslyUseFocusPseudoClass
            onClick={hideModal}
            /**
             * Hack: The footer will remove the margin to the save tags button
             * if the button if wrapped on another component.
             */
            css={{ marginRight: !hasNewValues ? theme.spacing.sm : 0 }}
          >
            {intl.formatMessage({
              defaultMessage: 'Cancel',
              description: 'Key-value tag editor modal > Manage Tag cancel button',
            })}
          </Button>
          {showPopoverMessage ? (
            <UnsavedTagPopoverTrigger formValues={formValues} isLoading={isLoading} onSaveTask={saveTags} />
          ) : (
            <LegacyTooltip
              title={
                !hasNewValues
                  ? intl.formatMessage({
                      defaultMessage: 'Please add or remove one or more tags before saving',
                      description: 'Key-value tag editor modal > Tag disabled message',
                    })
                  : undefined
              }
            >
              <Button
                componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_174"
                dangerouslyUseFocusPseudoClass
                disabled={!hasNewValues}
                loading={isLoading}
                type="primary"
                onClick={saveTags}
              >
                {intl.formatMessage({
                  defaultMessage: 'Save tags',
                  description: 'Key-value tag editor modal > Manage Tag save button',
                })}
              </Button>
            </LegacyTooltip>
          )}
        </RestoreAntDDefaultClsPrefix>
      }
    >
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        css={{ display: 'flex', alignItems: 'flex-end', gap: theme.spacing.md }}
      >
        <div css={{ minWidth: 0, display: 'flex', gap: theme.spacing.md, flex: 1 }}>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor="key">
              {intl.formatMessage({
                defaultMessage: 'Key',
                description: 'Key-value tag editor modal > Key input label',
              })}
            </FormUI.Label>
            <TagKeySelectDropdown
              allAvailableTags={allAvailableTags || []}
              control={form.control}
              onKeyChangeCallback={onKeyChangeCallback}
            />
          </div>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor="value">
              {valueRequired
                ? intl.formatMessage({
                    defaultMessage: 'Value',
                    description: 'Key-value tag editor modal > Value input label (required)',
                  })
                : intl.formatMessage({
                    defaultMessage: 'Value (optional)',
                    description: 'Key-value tag editor modal > Value input label',
                  })}
            </FormUI.Label>
            <RHFControlledComponents.Input
              componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_223"
              name="value"
              control={form.control}
              aria-label={
                valueRequired
                  ? intl.formatMessage({
                      defaultMessage: 'Value',
                      description: 'Key-value tag editor modal > Value input label (required)',
                    })
                  : intl.formatMessage({
                      defaultMessage: 'Value (optional)',
                      description: 'Key-value tag editor modal > Value input label',
                    })
              }
              placeholder={intl.formatMessage({
                defaultMessage: 'Type a value',
                description: 'Key-value tag editor modal > Value input placeholder',
              })}
            />
          </div>
        </div>
        <LegacyTooltip
          title={intl.formatMessage({
            defaultMessage: 'Add tag',
            description: 'Key-value tag editor modal > Add tag button',
          })}
        >
          <Button
            componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_248"
            htmlType="submit"
            aria-label={intl.formatMessage({
              defaultMessage: 'Add tag',
              description: 'Key-value tag editor modal > Add tag button',
            })}
          >
            <PlusIcon />
          </Button>
        </LegacyTooltip>
      </form>
      {errorMessage && <FormUI.Message type="error" message={errorMessage} />}
      <div
        css={{
          display: 'flex',
          rowGap: theme.spacing.xs,
          flexWrap: 'wrap',
          marginTop: theme.spacing.sm,
        }}
      >
        {Array.from(finalTags.values()).map((tag) => (
          <KeyValueTag isClosable tag={tag} onClose={() => handleTagDelete(tag)} key={tag.key} />
        ))}
      </div>
    </Modal>
  );

  return { EditTagsModal, showEditTagsModal, isLoading };
};

function UnsavedTagPopoverTrigger({
  isLoading,
  formValues,
  onSaveTask,
}: {
  isLoading: boolean;
  formValues: any;
  onSaveTask: () => void;
}) {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const tagKeyDisplay = `${truncate(formValues.key, { length: 20 }) || '_'}`;
  const tagValueDisplay = formValues.value ? `:${truncate(formValues.value, { length: 20 })}` : '';
  const fullTagDisplay = `${tagKeyDisplay}${tagValueDisplay}`;

  const shownText = intl.formatMessage(
    {
      defaultMessage: 'Are you sure you want to save and close without adding "{tag}"',
      description: 'Key-value tag editor modal > Unsaved tag message',
    },
    {
      tag: fullTagDisplay,
    },
  );
  return (
    <Popover.Root componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_309">
      <Popover.Trigger asChild>
        <Button
          componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_306"
          dangerouslyUseFocusPseudoClass
          loading={isLoading}
          type="primary"
        >
          {intl.formatMessage({
            defaultMessage: 'Save tags',
            description: 'Key-value tag editor modal > Manage Tag save button',
          })}
        </Button>
      </Popover.Trigger>
      <Popover.Content align="end" aria-label={shownText}>
        <Typography.Paragraph css={{ maxWidth: 400 }}>{shownText}</Typography.Paragraph>
        <Popover.Close asChild>
          <Button
            componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_316"
            onClick={onSaveTask}
          >
            {intl.formatMessage({
              defaultMessage: 'Yes, save and close',
              description: 'Key-value tag editor modal > Unsaved tag message > Yes, save and close button',
            })}
          </Button>
        </Popover.Close>
        <Popover.Close asChild>
          <Button
            componentId="codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_324"
            type="primary"
            css={{ marginLeft: theme.spacing.sm }}
          >
            {intl.formatMessage({
              defaultMessage: 'Cancel',
              description: 'Key-value tag editor modal > Unsaved tag message > cancel button',
            })}
          </Button>
        </Popover.Close>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
}
