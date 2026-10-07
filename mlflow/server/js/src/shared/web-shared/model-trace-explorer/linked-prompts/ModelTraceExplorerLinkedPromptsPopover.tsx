import {
  ApplyDesignSystemContextOverrides,
  Button,
  Popover,
  TextBoxIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import { useMemo } from 'react';

import type { ModelTraceInfoV3 } from '../ModelTrace.types';
import { Link, useParams } from '../RoutingUtils';
import { getLinkedPromptRoute, MLFLOW_LINKED_PROMPTS_TAG, parseLinkedPrompts } from './utils';

export interface ModelTraceExplorerLinkedPromptsPopoverProps {
  experimentId?: string;
  traceInfo?: ModelTraceInfoV3;
}

export const ModelTraceExplorerLinkedPromptsPopover = ({
  experimentId,
  traceInfo,
}: ModelTraceExplorerLinkedPromptsPopoverProps): JSX.Element | null => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId: experimentIdFromParams } = useParams();
  const linkedPrompts = useMemo(
    () => parseLinkedPrompts(traceInfo?.tags?.[MLFLOW_LINKED_PROMPTS_TAG]),
    [traceInfo?.tags],
  );
  const promptExperimentId =
    traceInfo?.trace_location.type === 'MLFLOW_EXPERIMENT'
      ? traceInfo.trace_location.mlflow_experiment.experiment_id
      : (experimentId ?? experimentIdFromParams);

  if (!promptExperimentId || linkedPrompts.length === 0) {
    return null;
  }

  const linkedPromptsLabel = intl.formatMessage(
    {
      defaultMessage: '{count, plural, one {# linked prompt} other {# linked prompts}}',
      description: 'Linked prompts count in the trace explorer',
    },
    { count: linkedPrompts.length },
  );
  const triggerLabel = intl.formatMessage(
    {
      defaultMessage: '{linkedPromptsLabel} in trace metadata',
      description: 'Accessible label for linked prompts in the trace explorer metadata',
    },
    { linkedPromptsLabel },
  );

  return (
    <ApplyDesignSystemContextOverrides getPopupContainer={() => document.body}>
      <Popover.Root componentId="shared.model-trace-explorer.linked-prompts-popover">
        <Popover.Trigger asChild>
          <Button
            componentId="shared.model-trace-explorer.linked-prompts-metadata"
            aria-label={triggerLabel}
            icon={<TextBoxIcon />}
            size="small"
            type="tertiary"
          >
            {linkedPromptsLabel}
          </Button>
        </Popover.Trigger>
        <Popover.Content align="end" minWidth={320} maxWidth={400}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Text bold>
              {intl.formatMessage({
                defaultMessage: 'Linked prompts',
                description: 'Title for the linked prompts popover in the trace explorer',
              })}
            </Typography.Text>
            <ul
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                listStyle: 'none',
                margin: 0,
                padding: `${theme.spacing.sm}px 0 0`,
                borderTop: `1px solid ${theme.colors.border}`,
                maxHeight: 240,
                overflowY: 'auto',
              }}
            >
              {linkedPrompts.map(({ name, version }) => (
                <li
                  key={`${name}-${version}`}
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.sm,
                    minWidth: 0,
                    whiteSpace: 'nowrap',
                  }}
                >
                  <Link
                    // prettier-ignore
                    componentId="shared.model-trace-explorer.linked-prompts-popover.prompt-link"
                    to={getLinkedPromptRoute({ experimentId: promptExperimentId, name, version })}
                    target="_blank"
                    rel="noreferrer"
                    css={{
                      flex: 1,
                      minWidth: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {name}
                  </Link>
                  <span css={{ flexShrink: 0, whiteSpace: 'nowrap' }}>
                    <Typography.Text color="secondary">
                      {intl.formatMessage(
                        {
                          defaultMessage: 'Version {version}',
                          description: 'Prompt version shown in the linked prompts popover',
                        },
                        { version },
                      )}
                    </Typography.Text>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Popover.Content>
      </Popover.Root>
    </ApplyDesignSystemContextOverrides>
  );
};
