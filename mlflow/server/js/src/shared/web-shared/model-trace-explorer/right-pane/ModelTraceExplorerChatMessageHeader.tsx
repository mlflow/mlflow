import {
  ChevronRightIcon,
  ChevronDownIcon,
  Typography,
  useDesignSystemTheme,
  Tooltip,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { ModelIconType } from '../ModelTrace.types';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';

const getRoleIcon = (role: string) => {
  switch (role) {
    case 'system':
      return <ModelTraceExplorerIcon type={ModelIconType.SYSTEM} />;
    case 'user':
      return <ModelTraceExplorerIcon type={ModelIconType.USER} />;
    case 'tool':
    case 'function':
      return <ModelTraceExplorerIcon type={ModelIconType.WRENCH} />;
    default:
      return <ModelTraceExplorerIcon type={ModelIconType.MODELS} />;
  }
};

const getRoleDisplayText = (message: ModelTraceChatMessage) => {
  switch (message.role) {
    case 'system':
      return (
        <FormattedMessage
          defaultMessage="System"
          description="Display text for the 'system' role in a GenAI chat message."
        />
      );
    case 'user':
      return (
        <FormattedMessage
          defaultMessage="User"
          description="Display text for the 'user' role in a GenAI chat message."
        />
      );
    case 'assistant':
      return (
        <FormattedMessage
          defaultMessage="Assistant"
          description="Display text for the 'assistant' role in a GenAI chat message."
        />
      );
    case 'tool':
      if (message.name) {
        return message.name;
      }
      return (
        <FormattedMessage
          defaultMessage="Tool"
          description="Display text for the 'tool' role in a GenAI chat message."
        />
      );
    case 'function':
      return (
        <FormattedMessage
          defaultMessage="Function"
          description="Display text for the 'function' role in a GenAI chat message."
        />
      );
    default:
      return message.role;
  }
};

export const ModelTraceExplorerChatMessageHeader = ({
  isExpandable,
  expanded,
  setExpanded,
  message,
}: {
  isExpandable: boolean;
  expanded: boolean;
  setExpanded: (expanded: boolean) => void;
  message: ModelTraceChatMessage;
}) => {
  const { theme } = useDesignSystemTheme();
  const hoverStyles = isExpandable
    ? {
        ':hover': {
          backgroundColor: theme.colors.actionIconBackgroundHover,
          cursor: 'pointer',
        },
      }
    : {};

  return (
    <div
      role="button"
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        padding: theme.spacing.sm,
        gap: theme.spacing.sm,
        ...hoverStyles,
      }}
      onClick={() => setExpanded(!expanded)}
    >
      {isExpandable && (expanded ? <ChevronDownIcon /> : <ChevronRightIcon />)}
      {getRoleIcon(message.role)}
      {message.tool_call_id ? (
        <Typography.Text
          color="secondary"
          css={{
            whiteSpace: 'nowrap',
            display: 'inline-flex',
            alignItems: 'center',
            flex: 1,
            minWidth: 0,
          }}
        >
          <FormattedMessage
            defaultMessage="{toolName} was called in {toolCallId}"
            description="A message that shows the tool call ID of a tool call chat message."
            values={{
              toolName: (
                <Typography.Text css={{ marginRight: theme.spacing.xs }} bold>
                  {getRoleDisplayText(message)}
                </Typography.Text>
              ),
              toolCallId: (
                <Tooltip componentId="test" content={message.tool_call_id}>
                  <div
                    css={{ display: 'inline-flex', flexShrink: 1, overflow: 'hidden', marginLeft: theme.spacing.xs }}
                  >
                    <Typography.Text css={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} code>
                      {message.tool_call_id}
                    </Typography.Text>
                  </div>
                </Tooltip>
              ),
            }}
          />
        </Typography.Text>
      ) : (
        <Typography.Text bold>{getRoleDisplayText(message)}</Typography.Text>
      )}
    </div>
  );
};
