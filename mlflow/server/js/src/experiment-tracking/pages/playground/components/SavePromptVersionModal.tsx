import {
  Alert,
  Checkbox,
  FormUI,
  Input,
  Modal,
  SaveIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useEffect, useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useCreateRegisteredPromptMutation } from '../../prompts/hooks/useCreateRegisteredPromptMutation';
import { PROMPT_MODEL_CONFIG_TAG_KEY, PROMPT_TYPE_CHAT, PROMPT_TYPE_TEXT } from '../../prompts/utils';
import type { ChatMessage, PlaygroundParams, PromptType, ResponseFormatType } from '../types';
import { getSaveableMessages, hasSaveableSettings, paramsToModelConfig } from '../promptVersionSave';

interface Props {
  visible: boolean;
  onCancel: () => void;
  messages: ChatMessage[];
  params: PlaygroundParams;
  responseFormatType: ResponseFormatType;
  responseFormatSchemaText: string;
  // Name of the prompt currently loaded into the playground, if any. When set,
  // the modal defaults to appending a new version to it.
  loadedPromptName?: string;
  // Registry type of the loaded prompt, so a text prompt that is still a single
  // user message can be saved back as text instead of being promoted to chat.
  loadedPromptType?: PromptType;
  onSaved: (result: { name: string; version: string; promptType: PromptType }) => void;
}

type Target = 'existing' | 'new';

const NAME_PATTERN = /^[a-zA-Z0-9_\-.]+$/;
const PREVIEW_CONTENT_CAP = 120;

const truncate = (s: string, cap: number) => (s.length > cap ? `${s.slice(0, cap)}…` : s);

export const SavePromptVersionModal = ({
  visible,
  onCancel,
  messages,
  params,
  responseFormatType,
  responseFormatSchemaText,
  loadedPromptName,
  loadedPromptType,
  onSaved,
}: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const queryClient = useQueryClient();

  const [target, setTarget] = useState<Target>(loadedPromptName ? 'existing' : 'new');
  const [newPromptName, setNewPromptName] = useState('');
  const [commitMessage, setCommitMessage] = useState('');
  const [includeSettings, setIncludeSettings] = useState(true);
  const [nameTouched, setNameTouched] = useState(false);

  const { mutate, error, isLoading, reset: resetMutation } = useCreateRegisteredPromptMutation();

  // Reset the form each time the modal is opened so it reflects the prompt
  // that is currently loaded (and clears any prior error/input).
  useEffect(() => {
    if (!visible) return;
    setTarget(loadedPromptName ? 'existing' : 'new');
    setNewPromptName('');
    setCommitMessage('');
    setIncludeSettings(true);
    setNameTouched(false);
    resetMutation();
  }, [visible, loadedPromptName, resetMutation]);

  const effectiveTarget: Target = loadedPromptName ? target : 'new';
  const saveableMessages = useMemo(() => getSaveableMessages(messages), [messages]);
  const settingsAvailable = hasSaveableSettings(params, responseFormatType, responseFormatSchemaText);

  const trimmedName = newPromptName.trim();
  const nameError = useMemo(() => {
    if (effectiveTarget !== 'new') return null;
    if (!trimmedName) {
      return intl.formatMessage({
        defaultMessage: 'Name is required',
        description: 'Validation error when no prompt name is provided in the playground save-prompt-version modal',
      });
    }
    if (!NAME_PATTERN.test(trimmedName)) {
      return intl.formatMessage({
        defaultMessage: 'Only alphanumeric characters, underscores, hyphens, and dots are allowed',
        description: 'Validation error for an invalid prompt name in the playground save-prompt-version modal',
      });
    }
    return null;
  }, [effectiveTarget, trimmedName, intl]);

  const targetName = effectiveTarget === 'existing' ? (loadedPromptName ?? '') : trimmedName;
  const hasMessages = saveableMessages.length > 0;
  const canSave = hasMessages && !nameError && targetName.length > 0 && !isLoading;

  const handleSave = () => {
    if (!canSave) {
      return;
    }
    const modelConfig = includeSettings ? paramsToModelConfig(params) : undefined;
    const modelConfigTags = modelConfig
      ? [{ key: PROMPT_MODEL_CONFIG_TAG_KEY, value: JSON.stringify(modelConfig) }]
      : [];
    const responseFormatJson =
      includeSettings && responseFormatType === 'json_schema' && responseFormatSchemaText.trim()
        ? responseFormatSchemaText.trim()
        : undefined;

    // Keep a loaded text prompt text-typed while it stays a single user message;
    // anything else (multiple turns, a system role) is saved as chat. Assistant
    // turns are already stripped by getSaveableMessages.
    const isSingleUserMessage = saveableMessages.length === 1 && saveableMessages[0].role === 'user';
    const promptType: PromptType =
      effectiveTarget === 'existing' && loadedPromptType === PROMPT_TYPE_TEXT && isSingleUserMessage
        ? PROMPT_TYPE_TEXT
        : PROMPT_TYPE_CHAT;
    const content =
      promptType === PROMPT_TYPE_TEXT ? (saveableMessages[0].content ?? '') : JSON.stringify(saveableMessages);

    mutate(
      {
        createPromptEntity: effectiveTarget === 'new',
        promptName: targetName,
        promptType,
        content,
        commitMessage: commitMessage.trim() || undefined,
        tags: modelConfigTags,
        responseFormatJson,
      },
      {
        onSuccess: (data) => {
          queryClient.invalidateQueries({ queryKey: ['prompts_list'] });
          queryClient.invalidateQueries({ queryKey: ['prompt_details', { promptName: targetName }] });
          onSaved({ name: targetName, version: data.version, promptType });
        },
      },
    );
  };

  return (
    <Modal
      componentId="mlflow.playground.save_prompt_version"
      visible={visible}
      onCancel={onCancel}
      title={
        <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <SaveIcon />
          <FormattedMessage
            defaultMessage="Save prompt to registry"
            description="Title of the save-prompt-version modal on the playground page"
          />
        </span>
      }
      okText={
        <FormattedMessage
          defaultMessage="Save version"
          description="Confirm-button label on the playground save-prompt-version modal"
        />
      }
      okButtonProps={{ disabled: !canSave, loading: isLoading }}
      onOk={handleSave}
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancel-button label on the playground save-prompt-version modal"
        />
      }
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Paragraph withoutMargins>
          <FormattedMessage
            defaultMessage="Save the current playground messages as a new prompt version in the registry."
            description="Intro paragraph at the top of the playground save-prompt-version modal"
          />
        </Typography.Paragraph>

        {error?.message && (
          <Alert
            componentId="mlflow.playground.save_prompt_version.error"
            type="error"
            closable={false}
            message={error.message}
          />
        )}

        {!hasMessages && (
          <Alert
            componentId="mlflow.playground.save_prompt_version.no_messages"
            type="warning"
            closable={false}
            message={
              <FormattedMessage
                defaultMessage="Add at least one non-empty message before saving."
                description="Warning shown in the playground save-prompt-version modal when there are no messages to save"
              />
            }
          />
        )}

        <div>
          <FormUI.Label>
            <FormattedMessage
              defaultMessage="Destination"
              description="Label for the destination selector in the playground save-prompt-version modal"
            />
          </FormUI.Label>
          {loadedPromptName ? (
            <SegmentedControlGroup
              componentId="mlflow.playground.save_prompt_version.target"
              name="mlflow.playground.save_prompt_version.target"
              value={target}
              onChange={(event) => setTarget(event.target.value as Target)}
            >
              <SegmentedControlButton value="existing">
                <FormattedMessage
                  defaultMessage="New version of {name}"
                  description="Option to save a new version of the loaded prompt in the playground save modal"
                  values={{ name: loadedPromptName }}
                />
              </SegmentedControlButton>
              <SegmentedControlButton value="new">
                <FormattedMessage
                  defaultMessage="New prompt"
                  description="Option to save the playground messages as a brand-new prompt in the save modal"
                />
              </SegmentedControlButton>
            </SegmentedControlGroup>
          ) : (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="A new prompt will be created in the registry."
                description="Note shown in the playground save modal when no prompt is loaded, so a new prompt is created"
              />
            </Typography.Text>
          )}
        </div>

        {effectiveTarget === 'new' && (
          <div>
            <FormUI.Label htmlFor="mlflow.playground.save_prompt_version.name">
              <FormattedMessage
                defaultMessage="Prompt name"
                description="Label for the new-prompt name input in the playground save-prompt-version modal"
              />
            </FormUI.Label>
            <Input
              componentId="mlflow.playground.save_prompt_version.name"
              id="mlflow.playground.save_prompt_version.name"
              value={newPromptName}
              onChange={(event: ChangeEvent<HTMLInputElement>) => setNewPromptName(event.target.value)}
              onBlur={() => setNameTouched(true)}
              placeholder={intl.formatMessage({
                defaultMessage: 'Provide a unique prompt name',
                description: 'Placeholder for the new-prompt name input in the playground save-prompt-version modal',
              })}
              validationState={nameTouched && nameError ? 'error' : undefined}
            />
            {nameTouched && nameError && <FormUI.Message type="error" message={nameError} />}
          </div>
        )}

        <div>
          <FormUI.Label htmlFor="mlflow.playground.save_prompt_version.commit_message">
            <FormattedMessage
              defaultMessage="Commit message (optional)"
              description="Label for the optional commit-message input in the playground save-prompt-version modal"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.playground.save_prompt_version.commit_message"
            id="mlflow.playground.save_prompt_version.commit_message"
            value={commitMessage}
            onChange={(event: ChangeEvent<HTMLInputElement>) => setCommitMessage(event.target.value)}
          />
        </div>

        {settingsAvailable && (
          <Checkbox
            componentId="mlflow.playground.save_prompt_version.include_settings"
            isChecked={includeSettings}
            onChange={(checked) => setIncludeSettings(checked)}
          >
            <FormattedMessage
              defaultMessage="Save model settings with this version"
              description="Label for the checkbox that stores playground model settings alongside the saved prompt version"
            />
          </Checkbox>
        )}

        {hasMessages && (
          <>
            <div css={{ borderTop: `1px solid ${theme.colors.border}` }} role="separator" aria-hidden="true" />
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.general.borderRadiusBase,
                padding: theme.spacing.md,
              }}
            >
              <Typography.Text
                size="sm"
                color="secondary"
                bold
                css={{ textTransform: 'uppercase', letterSpacing: 0.5 }}
              >
                <FormattedMessage
                  defaultMessage="Messages"
                  description="Header of the messages preview in the playground save-prompt-version modal"
                />
              </Typography.Text>
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.xs,
                  maxHeight: 240,
                  overflowY: 'auto',
                  paddingRight: theme.spacing.xs,
                }}
              >
                {saveableMessages.map((m, i) => (
                  <div key={`${m.role}-${i}`} css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'baseline' }}>
                    <Typography.Text bold css={{ minWidth: 72 }}>
                      {m.role}
                    </Typography.Text>
                    <Typography.Text color="secondary">
                      {truncate(m.content ?? '', PREVIEW_CONTENT_CAP)}
                    </Typography.Text>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};
