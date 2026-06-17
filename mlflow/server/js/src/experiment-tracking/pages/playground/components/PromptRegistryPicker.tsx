import {
  Alert,
  BookIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  Modal,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  MODEL_CONFIG_FIELD_LABELS,
  getModelConfigFromTags,
  getResponseFormatFromTags,
  isChatPrompt,
} from '../../prompts/utils';
import { usePromptDetailsQuery } from '../../prompts/hooks/usePromptDetailsQuery';
import { usePromptsListQuery } from '../../prompts/hooks/usePromptsListQuery';
import { buildLoadPayloadFromVersion } from '../promptVersionLoad';
import type { PromptLoadPayload } from '../promptVersionLoad';

export type { PromptLoadPayload } from '../promptVersionLoad';

interface Props {
  visible: boolean;
  onCancel: () => void;
  onLoad: (payload: PromptLoadPayload) => void;
}

const PREVIEW_CONTENT_CAP = 120;

const truncate = (s: string, cap: number) => (s.length > cap ? `${s.slice(0, cap)}…` : s);

const SETTINGS_FIELD_ORDER: Array<keyof typeof MODEL_CONFIG_FIELD_LABELS> = [
  'temperature',
  'max_tokens',
  'top_p',
  'top_k',
  'frequency_penalty',
  'presence_penalty',
  'stop_sequences',
];

export const PromptRegistryPicker = ({ visible, onCancel, onLoad }: Props) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

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

  const loadPayload = useMemo(
    () => (selectedVersionEntity ? buildLoadPayloadFromVersion(selectedVersionEntity) : null),
    [selectedVersionEntity],
  );

  const previewModelConfig = useMemo(
    () => (selectedVersionEntity ? getModelConfigFromTags(selectedVersionEntity.tags) : undefined),
    [selectedVersionEntity],
  );

  const previewHasResponseFormat = useMemo(() => {
    if (!selectedVersionEntity) return false;
    const raw = getResponseFormatFromTags(selectedVersionEntity.tags);
    return typeof raw === 'string' && raw.trim().length > 0;
  }, [selectedVersionEntity]);

  const handlePromptSelect = (name: string) => {
    setSelectedPromptName(name);
    setSelectedVersion(undefined);
  };

  const handleLoad = () => {
    if (!loadPayload) {
      return;
    }
    onLoad(loadPayload);
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
      visible={visible}
      onCancel={handleCancel}
      title={
        <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <BookIcon />
          <FormattedMessage
            defaultMessage="Load prompt from registry"
            description="Title of the prompt-registry picker modal on the playground page"
          />
        </span>
      }
      okText={
        <FormattedMessage
          defaultMessage="Load"
          description="Confirm-button label on the prompt-registry picker modal that loads the selected prompt version"
        />
      }
      okButtonProps={{ disabled: !loadPayload }}
      onOk={handleLoad}
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancel-button label on the prompt-registry picker modal"
        />
      }
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Paragraph withoutMargins>
          <FormattedMessage
            defaultMessage="Load a saved prompt to replace the current playground messages, and apply any settings stored with it."
            description="Intro paragraph at the top of the playground prompt-registry picker modal, explaining that loading replaces messages and applies stored settings"
          />
        </Typography.Paragraph>
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
        </div>

        <div>
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
        </div>

        {loadPayload && (
          <>
            <div css={{ borderTop: `1px solid ${theme.colors.border}` }} role="separator" aria-hidden="true" />
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.md,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.general.borderRadiusBase,
                padding: theme.spacing.md,
              }}
            >
              <Typography.Title
                level={3}
                withoutMargins
                css={{
                  borderBottom: `1px solid ${theme.colors.border}`,
                  paddingBottom: theme.spacing.xs,
                }}
              >
                <FormattedMessage
                  defaultMessage="Preview"
                  description="Title of the preview section in the playground prompt-registry picker modal"
                />
              </Typography.Title>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <Typography.Text
                  size="sm"
                  color="secondary"
                  bold
                  css={{ textTransform: 'uppercase', letterSpacing: 0.5 }}
                >
                  <FormattedMessage
                    defaultMessage="Messages"
                    description="Subsection header listing messages of the selected prompt version on the playground prompt-registry picker modal"
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
                  {loadPayload.messages.length === 0 ? (
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="(no messages)"
                        description="Placeholder shown in the playground prompt-registry preview when a prompt version contains no messages"
                      />
                    </Typography.Text>
                  ) : (
                    loadPayload.messages.map((m, i) => (
                      <div
                        key={`${m.role}-${i}`}
                        css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'baseline' }}
                      >
                        <Typography.Text bold css={{ minWidth: 72 }}>
                          {m.role}
                        </Typography.Text>
                        <Typography.Text color="secondary">{truncate(m.content, PREVIEW_CONTENT_CAP)}</Typography.Text>
                      </div>
                    ))
                  )}
                </div>
                {!isChatPrompt(selectedVersionEntity) && (
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
                )}
              </div>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <Typography.Text
                  size="sm"
                  color="secondary"
                  bold
                  css={{ textTransform: 'uppercase', letterSpacing: 0.5 }}
                >
                  <FormattedMessage
                    defaultMessage="Settings"
                    description="Subsection header listing model settings stored with the selected prompt version on the playground prompt-registry picker modal"
                  />
                </Typography.Text>
                {loadPayload.settings === null ? (
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="No settings stored with this version"
                      description="Muted note shown in the playground prompt-registry preview when the selected prompt version has no model config or response format stored"
                    />
                  </Typography.Text>
                ) : (
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    {SETTINGS_FIELD_ORDER.map((key) => {
                      const value = previewModelConfig?.[key];
                      if (value === undefined) return null;
                      if (Array.isArray(value)) {
                        if (value.length === 0) return null;
                        return (
                          <div key={key} css={{ display: 'flex', gap: theme.spacing.sm }}>
                            <Typography.Text bold>{MODEL_CONFIG_FIELD_LABELS[key]}:</Typography.Text>
                            <Typography.Text color="secondary">{value.join(', ')}</Typography.Text>
                          </div>
                        );
                      }
                      return (
                        <div key={key} css={{ display: 'flex', gap: theme.spacing.sm }}>
                          <Typography.Text bold>{MODEL_CONFIG_FIELD_LABELS[key]}:</Typography.Text>
                          <Typography.Text color="secondary">{String(value)}</Typography.Text>
                        </div>
                      );
                    })}
                    {previewHasResponseFormat && (
                      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
                        <Typography.Text bold>
                          <FormattedMessage
                            defaultMessage="Response format:"
                            description="Label preceding the response-format value in the playground prompt-registry preview"
                          />
                        </Typography.Text>
                        <Typography.Text color="secondary">
                          <FormattedMessage
                            defaultMessage="JSON schema"
                            description="Value shown next to the response-format label when the selected prompt version stores a JSON-schema response format"
                          />
                        </Typography.Text>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};
