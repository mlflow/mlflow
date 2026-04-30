import {
  Alert,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  Modal,
  Spacer,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { usePromptDetailsQuery } from '../../prompts/hooks/usePromptDetailsQuery';
import { usePromptsListQuery } from '../../prompts/hooks/usePromptsListQuery';
import { getChatPromptMessagesFromValue, getPromptContentTagValue, isChatPrompt } from '../../prompts/utils';
import type { ChatMessage } from '../types';

interface Props {
  visible: boolean;
  onCancel: () => void;
  onLoad: (messages: ChatMessage[]) => void;
}

const COMPONENT_ID = 'mlflow.playground.prompt_registry_picker';

const buildMessagesFromVersion = (value: string | undefined, isChat: boolean): ChatMessage[] => {
  if (!value) {
    return [];
  }
  if (!isChat) {
    return [{ role: 'user', content: value }];
  }
  const parsed = getChatPromptMessagesFromValue(value);
  if (!parsed) {
    return [{ role: 'user', content: value }];
  }
  return parsed.map((msg) => ({
    role: (msg.role as ChatMessage['role']) ?? 'user',
    content: msg.content,
  }));
};

export const PromptRegistryPicker = ({ visible, onCancel, onLoad }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [selectedPromptName, setSelectedPromptName] = useState<string | undefined>();
  const [selectedVersion, setSelectedVersion] = useState<string | undefined>();

  const { data: prompts, isLoading: isPromptsLoading, error: promptsError } = usePromptsListQuery();

  const {
    data: promptDetails,
    isLoading: isDetailsLoading,
    error: detailsError,
  } = usePromptDetailsQuery({ promptName: selectedPromptName ?? '' }, { enabled: Boolean(selectedPromptName) });

  const versions = useMemo(() => promptDetails?.versions ?? [], [promptDetails?.versions]);

  const selectedVersionEntity = useMemo(
    () => versions.find((v) => v.version === selectedVersion),
    [versions, selectedVersion],
  );

  const handleLoad = () => {
    if (!selectedVersionEntity) {
      return;
    }
    const value = getPromptContentTagValue(selectedVersionEntity);
    const messages = buildMessagesFromVersion(value, isChatPrompt(selectedVersionEntity));
    onLoad(messages);
  };

  const reset = () => {
    setSelectedPromptName(undefined);
    setSelectedVersion(undefined);
  };

  const handleCancel = () => {
    reset();
    onCancel();
  };

  return (
    <Modal
      componentId={`${COMPONENT_ID}.modal`}
      title={
        <FormattedMessage
          defaultMessage="Load prompt from registry"
          description="Title for the modal that loads a registered prompt into the playground"
        />
      }
      visible={visible}
      onCancel={handleCancel}
      okText={
        <FormattedMessage
          defaultMessage="Load"
          description="Confirm button label for loading a registered prompt into the playground"
        />
      }
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancel button label for the load-prompt modal on the playground page"
        />
      }
      okButtonProps={{ disabled: !selectedVersionEntity }}
      onOk={handleLoad}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {promptsError && (
          <Alert type="error" message={promptsError.message} componentId={`${COMPONENT_ID}.error`} closable={false} />
        )}
        <div>
          <FormUI.Label htmlFor={`${COMPONENT_ID}.prompt`}>
            <FormattedMessage
              defaultMessage="Prompt"
              description="Label for the prompt selector in the playground load-prompt modal"
            />
          </FormUI.Label>
          <DialogCombobox
            componentId={`${COMPONENT_ID}.prompt`}
            label={intl.formatMessage({
              defaultMessage: 'Prompt',
              description: 'Label for the prompt selector in the playground load-prompt modal',
            })}
            modal={false}
            value={selectedPromptName ? [selectedPromptName] : undefined}
          >
            <DialogComboboxTrigger
              id={`${COMPONENT_ID}.prompt`}
              css={{ width: '100%' }}
              allowClear
              placeholder={intl.formatMessage({
                defaultMessage: 'Select a prompt',
                description: 'Placeholder for the prompt selector in the playground load-prompt modal',
              })}
              withInlineLabel={false}
              onClear={reset}
            />
            <DialogComboboxContent loading={isPromptsLoading} maxHeight={300} matchTriggerWidth>
              {!isPromptsLoading && (
                <DialogComboboxOptionList>
                  <DialogComboboxOptionListSearch autoFocus>
                    {(prompts ?? []).map((prompt) => (
                      <DialogComboboxOptionListSelectItem
                        value={prompt.name}
                        key={prompt.name}
                        onChange={(name) => {
                          setSelectedPromptName(name);
                          setSelectedVersion(undefined);
                        }}
                        checked={selectedPromptName === prompt.name}
                      >
                        {prompt.name}
                      </DialogComboboxOptionListSelectItem>
                    ))}
                  </DialogComboboxOptionListSearch>
                </DialogComboboxOptionList>
              )}
            </DialogComboboxContent>
          </DialogCombobox>
        </div>

        {selectedPromptName && (
          <div>
            <FormUI.Label htmlFor={`${COMPONENT_ID}.version`}>
              <FormattedMessage
                defaultMessage="Version"
                description="Label for the version selector in the playground load-prompt modal"
              />
            </FormUI.Label>
            {isDetailsLoading ? (
              <Spinner />
            ) : detailsError ? (
              <Alert
                type="error"
                message={detailsError.message}
                componentId={`${COMPONENT_ID}.details_error`}
                closable={false}
              />
            ) : (
              <DialogCombobox
                componentId={`${COMPONENT_ID}.version`}
                label={intl.formatMessage({
                  defaultMessage: 'Version',
                  description: 'Label for the version selector in the playground load-prompt modal',
                })}
                modal={false}
                value={selectedVersion ? [selectedVersion] : undefined}
              >
                <DialogComboboxTrigger
                  id={`${COMPONENT_ID}.version`}
                  css={{ width: '100%' }}
                  allowClear
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Select a version',
                    description: 'Placeholder for the version selector in the playground load-prompt modal',
                  })}
                  withInlineLabel={false}
                  onClear={() => setSelectedVersion(undefined)}
                />
                <DialogComboboxContent maxHeight={300} matchTriggerWidth>
                  <DialogComboboxOptionList>
                    {versions.map((version) => (
                      <DialogComboboxOptionListSelectItem
                        value={version.version ?? ''}
                        key={version.version}
                        onChange={(value) => setSelectedVersion(value)}
                        checked={selectedVersion === version.version}
                      >
                        {version.version}
                      </DialogComboboxOptionListSelectItem>
                    ))}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
            )}
          </div>
        )}

        <Spacer size="sm" />
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Variables in the prompt (e.g. {example}) will be loaded as-is. Edit the prompt before submitting to fill them in."
            description="Helper text on the playground load-prompt modal explaining variable handling"
            values={{ example: '{{name}}' }}
          />
        </Typography.Hint>
      </div>
    </Modal>
  );
};
