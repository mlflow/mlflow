import { FormUI, InfoSmallIcon, Input, Popover, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { PlaygroundParams } from '../types';

interface Props {
  value: PlaygroundParams;
  onChange: (next: PlaygroundParams) => void;
}

const parseNumber = (raw: string): number | undefined => {
  if (raw.trim() === '') {
    return undefined;
  }
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const formatValue = (value: number | undefined): string => (value === undefined ? '' : String(value));

export const ParametersPanel = ({ value, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const handleField = (field: keyof PlaygroundParams) => (event: ChangeEvent<HTMLInputElement>) => {
    onChange({ ...value, [field]: parseNumber(event.target.value) });
  };

  return (
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
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Title level={4} withoutMargins>
          <FormattedMessage
            defaultMessage="Parameters"
            description="Section header for the model sampling parameters panel on the playground page"
          />
        </Typography.Title>
        <Popover.Root componentId="mlflow.playground.params.help">
          <Popover.Trigger
            aria-label={intl.formatMessage({
              defaultMessage: 'About sampling parameters',
              description: 'Aria label for the info popover next to the playground parameters header',
            })}
            css={{ border: 0, background: 'none', padding: 0, display: 'inline-flex', cursor: 'pointer' }}
          >
            <InfoSmallIcon />
          </Popover.Trigger>
          <Popover.Content align="start" css={{ maxWidth: 320 }}>
            <Typography.Paragraph withoutMargins>
              <FormattedMessage
                defaultMessage="Leave a field blank to use the provider's default."
                description="Help text in the playground parameters panel popover"
              />
            </Typography.Paragraph>
            <Popover.Arrow />
          </Popover.Content>
        </Popover.Root>
      </div>

      <FormUI.Label htmlFor="mlflow.playground.params.temperature">
        <FormattedMessage
          defaultMessage="Temperature"
          description="Label for the temperature input on the playground parameters panel"
        />
      </FormUI.Label>
      <Input
        componentId="mlflow.playground.params.temperature"
        id="mlflow.playground.params.temperature"
        type="number"
        min={0}
        max={2}
        step={0.1}
        value={formatValue(value.temperature)}
        placeholder={intl.formatMessage({
          defaultMessage: 'e.g. 0.7',
          description: 'Placeholder for the temperature input on the playground parameters panel',
        })}
        onChange={handleField('temperature')}
      />

      <FormUI.Label htmlFor="mlflow.playground.params.max_tokens">
        <FormattedMessage
          defaultMessage="Max tokens"
          description="Label for the max-tokens input on the playground parameters panel"
        />
      </FormUI.Label>
      <Input
        componentId="mlflow.playground.params.max_tokens"
        id="mlflow.playground.params.max_tokens"
        type="number"
        min={1}
        step={1}
        value={formatValue(value.max_tokens)}
        placeholder={intl.formatMessage({
          defaultMessage: 'e.g. 512',
          description: 'Placeholder for the max-tokens input on the playground parameters panel',
        })}
        onChange={handleField('max_tokens')}
      />

      <FormUI.Label htmlFor="mlflow.playground.params.top_p">
        <FormattedMessage
          defaultMessage="Top P"
          description="Label for the top-p input on the playground parameters panel"
        />
      </FormUI.Label>
      <Input
        componentId="mlflow.playground.params.top_p"
        id="mlflow.playground.params.top_p"
        type="number"
        min={0}
        max={1}
        step={0.05}
        value={formatValue(value.top_p)}
        placeholder={intl.formatMessage({
          defaultMessage: 'e.g. 1.0',
          description: 'Placeholder for the top-p input on the playground parameters panel',
        })}
        onChange={handleField('top_p')}
      />
    </div>
  );
};
