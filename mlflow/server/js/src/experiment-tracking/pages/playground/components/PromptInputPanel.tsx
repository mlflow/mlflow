import {
  Button,
  CloseIcon,
  FormUI,
  InfoSmallIcon,
  Input,
  PlusIcon,
  Popover,
  SegmentedControlButton,
  SegmentedControlGroup,
  Space,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChangeEvent } from 'react';
import type { ChatMessage, ChatRole } from '../types';

const { TextArea } = Input;

const COMPONENT_ID = 'mlflow.playground.prompt_input';

const ROLE_OPTIONS: ChatRole[] = ['system', 'user', 'assistant'];

interface Props {
  messages: ChatMessage[];
  onChange: (messages: ChatMessage[]) => void;
}

export const PromptInputPanel = ({ messages, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const updateMessage = (index: number, patch: Partial<ChatMessage>) => {
    onChange(messages.map((msg, i) => (i === index ? { ...msg, ...patch } : msg)));
  };

  const removeMessage = (index: number) => {
    onChange(messages.filter((_, i) => i !== index));
  };

  const addMessage = () => {
    onChange([...messages, { role: 'user', content: '' }]);
  };

  return (
    <Space direction="vertical" size="small" css={{ width: '100%' }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Title level={4} withoutMargins>
          <FormattedMessage
            defaultMessage="Messages"
            description="Section header for the chat input on the playground page"
          />
        </Typography.Title>
        <Popover.Root componentId="mlflow.playground.prompt_input.role_help">
          <Popover.Trigger
            aria-label={intl.formatMessage({
              defaultMessage: 'About message roles',
              description: 'Aria label for the info popover that explains chat roles on the playground page',
            })}
            css={{ border: 0, background: 'none', padding: 0, display: 'inline-flex', cursor: 'pointer' }}
          >
            <InfoSmallIcon />
          </Popover.Trigger>
          <Popover.Content align="start" css={{ maxWidth: 360 }}>
            <Typography.Paragraph withoutMargins>
              <FormattedMessage
                defaultMessage="Each message has a role that tells the model who is speaking:"
                description="Intro line for the playground role-selector help popover"
              />
            </Typography.Paragraph>
            <ul css={{ margin: `${theme.spacing.xs}px 0 0`, paddingLeft: theme.spacing.lg }}>
              <li>
                <Typography.Text bold>system</Typography.Text>{' '}
                <FormattedMessage
                  defaultMessage="— instructions or persona for the model (e.g. {example})."
                  description="Description of the system role in the playground role-selector help popover"
                  values={{ example: <Typography.Text code>You are a helpful assistant</Typography.Text> }}
                />
              </li>
              <li>
                <Typography.Text bold>user</Typography.Text>{' '}
                <FormattedMessage
                  defaultMessage="— what the human is asking. This is the typical input."
                  description="Description of the user role in the playground role-selector help popover"
                />
              </li>
              <li>
                <Typography.Text bold>assistant</Typography.Text>{' '}
                <FormattedMessage
                  defaultMessage="— what the model has previously said. Useful for priming with example responses."
                  description="Description of the assistant role in the playground role-selector help popover"
                />
              </li>
            </ul>
            <Popover.Arrow />
          </Popover.Content>
        </Popover.Root>
      </div>

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
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.general.borderRadiusBase,
            padding: theme.spacing.sm,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
          data-testid={`mlflow.playground.prompt_input.message.${message.role}`}
        >
          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <SegmentedControlGroup
              componentId="mlflow.playground.prompt_input.role"
              name={`${COMPONENT_ID}.role.${index}`}
              size="small"
              value={message.role}
              onChange={(event) => updateMessage(index, { role: event.target.value as ChatRole })}
            >
              {ROLE_OPTIONS.map((role) => (
                <SegmentedControlButton key={role} value={role}>
                  {role}
                </SegmentedControlButton>
              ))}
            </SegmentedControlGroup>
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
          </div>
          <FormUI.Label htmlFor={`${COMPONENT_ID}.content.${index}`} css={{ display: 'none' }}>
            <FormattedMessage
              defaultMessage="Message content"
              description="Hidden label for the message content textarea on the playground page"
            />
          </FormUI.Label>
          <TextArea
            componentId="mlflow.playground.prompt_input.content"
            id={`${COMPONENT_ID}.content.${index}`}
            value={message.content}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
              updateMessage(index, { content: event.target.value })
            }
            autoSize={{ minRows: 3, maxRows: 12 }}
            placeholder={intl.formatMessage({
              defaultMessage: 'Type a message',
              description: 'Placeholder for a message textarea on the playground page',
            })}
          />
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
