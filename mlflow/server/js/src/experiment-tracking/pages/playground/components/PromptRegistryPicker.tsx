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
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { getChatPromptMessagesFromValue, getPromptContentTagValue, isChatPrompt } from '../../prompts/utils';
import { usePromptDetailsQuery } from '../../prompts/hooks/usePromptDetailsQuery';
import { usePromptsListQuery } from '../../prompts/hooks/usePromptsListQuery';
import type { RegisteredPromptVersion } from '../../prompts/types';
import type { ChatMessage, ChatRole } from '../types';

interface Props {
  visible: boolean;
  onCancel: () => void;
  onLoad: (messages: ChatMessage[]) => void;
}

const KNOWN_ROLES: ChatRole[] = ['system', 'user', 'assistant'];

const isKnownRole = (role: string): role is ChatRole => (KNOWN_ROLES as string[]).includes(role);

const buildMessagesFromVersion = (version: RegisteredPromptVersion): ChatMessage[] => {
  const tagValue = getPromptContentTagValue(version) ?? '';

  if (isChatPrompt(version)) {
    const parsed = getChatPromptMessagesFromValue(tagValue);
    if (!parsed) {
      return [];
    }
    return parsed.map((message) => ({
      role: isKnownRole(message.role) ? message.role : 'user',
      content: message.content,
    }));
  }

  return [{ role: 'user', content: tagValue }];
};

export const PromptRegistryPicker = ({ visible, onCancel, onLoad }: Props) => {
  const intl = useIntl();

  const [selectedPromptName, setSelectedPromptName] = useState<string | undefined>(undefined);
  const [selectedVersion, setSelectedVersion] = useState<string | undefined>(undefined);

  const { data: prompts, isLoading: isPromptsLoading, error: promptsError } = usePromptsListQuery();
  const {
    data: details,
    isLoading: isDetailsLoading,
    error: detailsError,
  } = usePromptDetailsQuery({ promptName: selectedPromptName ?? '' }, { enabled: Boolean(selectedPromptName) });

  const versions = useMemo(() => details?.versions ?? [], [details]);

  const selectedVersionEntity = useMemo(
    () => versions.find((v) => v.version === selectedVersion),
    [versions, selectedVersion],
  );

  const handlePromptSelect = (name: string) => {
    setSelectedPromptName(name);
    setSelectedVersion(undefined);
  };

  const handleLoad = () => {
    if (!selectedVersionEntity) {
      return;
    }
    onLoad(buildMessagesFromVersion(selectedVersionEntity));
    setSelectedPromptName(undefined);
    setSelectedVersion(undefined);
  };

  const handleCancel = () => {
    setSelectedPromptName(undefined);
    setSelectedVersion(undefined);
    onCancel();
  };

  return (
    <Modal
      componentId="mlflow.playground.prompt_registry_picker"
      title={
        <FormattedMessage
          defaultMessage="Load prompt from registry"
          description="Title of the prompt-registry picker modal on the playground page"
        />
      }
      visible={visible}
      onCancel={handleCancel}
      onOk={handleLoad}
      okText={
        <FormattedMessage
          defaultMessage="Load"
          description="Confirm-button label on the prompt-registry picker modal that loads the selected prompt version"
        />
      }
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancel-button label on the prompt-registry picker modal"
        />
      }
      okButtonProps={{ disabled: !selectedVersionEntity }}
    >
      <div>
        <FormUI.Label htmlFor="mlflow.playground.prompt_registry_picker.prompt">
          <FormattedMessage
            defaultMessage="Prompt"
            description="Label for the prompt picker on the playground prompt-registry modal"
          />
        </FormUI.Label>
        <DialogCombobox
          componentId="mlflow.playground.prompt_registry_picker.prompt"
          label={intl.formatMessage({
            defaultMessage: 'Prompt',
            description: 'Label for the prompt picker on the playground prompt-registry modal',
          })}
          modal={false}
          value={selectedPromptName ? [selectedPromptName] : undefined}
        >
          <DialogComboboxTrigger
            id="mlflow.playground.prompt_registry_picker.prompt"
            css={{ width: '100%' }}
            allowClear
            placeholder={intl.formatMessage({
              defaultMessage: 'Select a prompt',
              description: 'Placeholder for the prompt picker on the playground prompt-registry modal',
            })}
            withInlineLabel={false}
            onClear={() => handlePromptSelect('')}
          />
          <DialogComboboxContent loading={isPromptsLoading} maxHeight={320} matchTriggerWidth>
            {!isPromptsLoading && (
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSearch autoFocus>
                  {(prompts ?? []).map((prompt) => (
                    <DialogComboboxOptionListSelectItem
                      key={prompt.name}
                      value={prompt.name}
                      onChange={(name) => handlePromptSelect(name)}
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
        {promptsError && <FormUI.Message type="error" message={promptsError.message} />}

        <Spacer size="md" />

        <FormUI.Label htmlFor="mlflow.playground.prompt_registry_picker.version">
          <FormattedMessage
            defaultMessage="Version"
            description="Label for the version picker on the playground prompt-registry modal"
          />
        </FormUI.Label>
        <DialogCombobox
          componentId="mlflow.playground.prompt_registry_picker.version"
          label={intl.formatMessage({
            defaultMessage: 'Version',
            description: 'Label for the version picker on the playground prompt-registry modal',
          })}
          modal={false}
          value={selectedVersion ? [selectedVersion] : undefined}
        >
          <DialogComboboxTrigger
            id="mlflow.playground.prompt_registry_picker.version"
            css={{ width: '100%' }}
            allowClear
            disabled={!selectedPromptName || isDetailsLoading}
            placeholder={intl.formatMessage({
              defaultMessage: 'Select a version',
              description: 'Placeholder for the version picker on the playground prompt-registry modal',
            })}
            withInlineLabel={false}
            onClear={() => setSelectedVersion(undefined)}
          />
          <DialogComboboxContent loading={isDetailsLoading} maxHeight={320} matchTriggerWidth>
            {!isDetailsLoading && (
              <DialogComboboxOptionList>
                {versions.map((version) => (
                  <DialogComboboxOptionListSelectItem
                    key={version.version}
                    value={version.version}
                    onChange={(value) => setSelectedVersion(value)}
                    checked={selectedVersion === version.version}
                  >
                    {`v${version.version}`}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionList>
            )}
          </DialogComboboxContent>
        </DialogCombobox>
        {detailsError && <FormUI.Message type="error" message={detailsError.message} />}

        {selectedVersionEntity && !isChatPrompt(selectedVersionEntity) && (
          <>
            <Spacer size="sm" />
            <Alert
              componentId="mlflow.playground.prompt_registry_picker.text_prompt_hint"
              type="info"
              closable={false}
              message={
                <FormattedMessage
                  defaultMessage="This is a text-typed prompt. It will load as a single user message."
                  description="Info alert shown when a text-typed prompt is selected on the playground prompt-registry modal"
                />
              }
            />
          </>
        )}
      </div>
    </Modal>
  );
};
