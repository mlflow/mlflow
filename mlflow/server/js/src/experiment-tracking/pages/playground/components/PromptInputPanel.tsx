import {
  Button,
  ChevronDownIcon,
  CloseIcon,
  DropdownMenu,
  InfoIcon,
  Input,
  PlusIcon,
  Popover,
  Space,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChangeEvent } from 'react';
import type { ChatRole, ConversationMessage } from '../types';

const { TextArea } = Input;

const COMPONENT_ID = 'mlflow.playground.prompt_input';

const ROLE_OPTIONS: ChatRole[] = ['system', 'user', 'assistant'];

const roleLabel = (role: ChatRole): string => role[0].toUpperCase() + role.slice(1);

const getNamedToolCalls = (message: ConversationMessage) =>
  message.tool_calls?.filter((toolCall) => Boolean(toolCall.function?.name)) ?? [];

interface Props {
  messages: ConversationMessage[];
  onChange: (messages: ConversationMessage[]) => void;
}

export const PromptInputPanel = ({ messages, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Shared grey-card styling reused by every assistant card (the text card and each
  // per-tool-call card) so they render as visually-identical sibling boxes.
  const cardStyles = {
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.general.borderRadiusBase,
    padding: theme.spacing.md,
    backgroundColor: theme.colors.backgroundSecondary,
    overflow: 'auto',
  } as const;

  const updateMessage = (index: number, patch: Partial<ConversationMessage>) => {
    onChange(messages.map((msg, i) => (i === index ? { ...msg, ...patch } : msg)));
  };

  const removeMessage = (index: number) => {
    onChange(messages.filter((_, i) => i !== index));
  };

  const addMessage = () => {
    onChange([...messages, { role: 'user', content: '' }]);
  };

  const removeButton = (index: number) => (
    <Button
      componentId="mlflow.playground.prompt_input.remove"
      size="small"
      icon={<CloseIcon />}
      aria-label={intl.formatMessage({
        defaultMessage: 'Remove message',
        description: 'Aria label for the button that removes a message from the playground messages panel',
      })}
      onClick={() => removeMessage(index)}
    />
  );

  return (
    <Space direction="vertical" size="middle" css={{ width: '100%' }}>
      {messages.length === 0 && (
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Add a message to send to the endpoint."
            description="Empty-state hint for the playground messages panel"
          />
        </Typography.Hint>
      )}

      {messages.map((message, index) => (
        <div
          key={index}
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
          data-testid={`mlflow.playground.prompt_input.message.${message.role}`}
        >
          {message.role === 'assistant' ? (
            <>
              <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography.Text
                  css={{
                    color: theme.colors.textSecondary,
                    fontSize: theme.typography.fontSizeSm,
                  }}
                >
                  {roleLabel(message.role)}
                </Typography.Text>
                {removeButton(index)}
              </div>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                {message.content ? (
                  <div css={cardStyles} data-testid="mlflow.playground.assistant.text_card">
                    <GenAIMarkdownRenderer>{message.content}</GenAIMarkdownRenderer>
                  </div>
                ) : null}
                {getNamedToolCalls(message).map((toolCall, toolCallIndex) => (
                  <div
                    key={toolCall.id ?? toolCallIndex}
                    css={cardStyles}
                    data-testid="mlflow.playground.assistant.tool_call_card"
                  >
                    <Typography.Text bold>
                      <strong>{toolCall.function?.name}</strong>
                    </Typography.Text>
                    {toolCall.function?.arguments && (
                      <GenAIMarkdownRenderer>{toolCall.function.arguments}</GenAIMarkdownRenderer>
                    )}
                  </div>
                ))}
                {!message.content && getNamedToolCalls(message).length === 0 && (
                  <div css={cardStyles} data-testid="mlflow.playground.assistant.text_card">
                    <GenAIMarkdownRenderer>(no text content)</GenAIMarkdownRenderer>
                  </div>
                )}
              </div>
              {message.usage && (
                <Space direction="vertical" size={0}>
                  <Typography.Hint>
                    <FormattedMessage
                      defaultMessage="Tokens — input: {input}, output: {output}, total: {total}"
                      description="Token usage footer rendered below each assistant reply on the playground page"
                      values={{
                        input: message.usage.prompt_tokens ?? '—',
                        output: message.usage.completion_tokens ?? '—',
                        total: message.usage.total_tokens ?? '—',
                      }}
                    />
                  </Typography.Hint>
                </Space>
              )}
            </>
          ) : (
            <>
              <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <DropdownMenu.Root>
                  <DropdownMenu.Trigger asChild>
                    <button
                      type="button"
                      aria-label={intl.formatMessage({
                        defaultMessage: 'Change message role',
                        description:
                          'Aria label for the dropdown that swaps a chat message role on the playground page',
                      })}
                      css={{
                        border: 0,
                        background: 'transparent',
                        padding: 0,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: theme.spacing.xs,
                        cursor: 'pointer',
                        color: theme.colors.textSecondary,
                        fontSize: theme.typography.fontSizeSm,
                        '&:hover': { color: theme.colors.textPrimary },
                      }}
                    >
                      {roleLabel(message.role)}
                      <ChevronDownIcon css={{ fontSize: theme.typography.fontSizeSm }} />
                    </button>
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Content align="start">
                    <DropdownMenu.RadioGroup
                      componentId={`${COMPONENT_ID}.role.group`}
                      value={message.role}
                      onValueChange={(role) => updateMessage(index, { role: role as ChatRole })}
                    >
                      {ROLE_OPTIONS.map((role) => (
                        <DropdownMenu.RadioItem key={role} value={role}>
                          <DropdownMenu.ItemIndicator />
                          {roleLabel(role)}
                        </DropdownMenu.RadioItem>
                      ))}
                    </DropdownMenu.RadioGroup>
                  </DropdownMenu.Content>
                </DropdownMenu.Root>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <Popover.Root componentId="mlflow.playground.prompt_input.role_help">
                    <Popover.Trigger
                      aria-label={intl.formatMessage({
                        defaultMessage: 'About message roles',
                        description: 'Aria label for the info popover that explains chat roles on the playground page',
                      })}
                      css={{
                        border: 0,
                        background: 'none',
                        padding: 0,
                        display: 'inline-flex',
                        cursor: 'pointer',
                        color: theme.colors.textSecondary,
                        '&:hover': { color: theme.colors.textPrimary },
                      }}
                    >
                      <InfoIcon />
                    </Popover.Trigger>
                    <Popover.Content align="end" css={{ maxWidth: 360 }}>
                      <Typography.Paragraph withoutMargins>
                        <FormattedMessage
                          defaultMessage="Each message has a role that tells the model who is speaking:"
                          description="Intro line for the playground role-selector help popover"
                        />
                      </Typography.Paragraph>
                      <ul css={{ margin: `${theme.spacing.xs}px 0 0`, paddingLeft: theme.spacing.lg }}>
                        <li>
                          <Typography.Text bold>System</Typography.Text>{' '}
                          <FormattedMessage
                            defaultMessage="— instructions or persona for the model (e.g. {example})."
                            description="Description of the system role in the playground role-selector help popover"
                            values={{ example: <Typography.Text code>You are a helpful assistant</Typography.Text> }}
                          />
                        </li>
                        <li>
                          <Typography.Text bold>User</Typography.Text>{' '}
                          <FormattedMessage
                            defaultMessage="— what the human is asking. This is the typical input."
                            description="Description of the user role in the playground role-selector help popover"
                          />
                        </li>
                        <li>
                          <Typography.Text bold>Assistant</Typography.Text>{' '}
                          <FormattedMessage
                            defaultMessage="— what the model has previously said. Useful for priming with example responses."
                            description="Description of the assistant role in the playground role-selector help popover"
                          />
                        </li>
                      </ul>
                      <Popover.Arrow />
                    </Popover.Content>
                  </Popover.Root>
                  {removeButton(index)}
                </div>
              </div>
              <TextArea
                componentId="mlflow.playground.prompt_input.content"
                value={message.content ?? ''}
                onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
                  updateMessage(index, { content: event.target.value })
                }
                autoSize={{ minRows: 3, maxRows: 12 }}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Type a message',
                  description: 'Placeholder for a message textarea on the playground page',
                })}
              />
            </>
          )}
        </div>
      ))}

      <Button componentId={`${COMPONENT_ID}.add`} icon={<PlusIcon />} onClick={addMessage}>
        <FormattedMessage
          defaultMessage="Add message"
          description="Label for the button that appends a new chat message on the playground page"
        />
      </Button>
    </Space>
  );
};
