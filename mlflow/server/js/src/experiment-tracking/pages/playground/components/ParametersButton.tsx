import { Button, Drawer, Typography, WrenchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { PlaygroundParams } from '../types';
import { ParametersForm } from './ParametersForm';

interface Props {
  value: PlaygroundParams;
  onChange: (next: PlaygroundParams) => void;
}

export const ParametersButton = ({ value, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <Drawer.Root>
      <Drawer.Trigger>
        <Button
          componentId="mlflow.playground.params.drawer.trigger"
          icon={<WrenchIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Open model parameters',
            description: 'Aria label for the wrench button that opens the playground parameters drawer',
          })}
        >
          <FormattedMessage
            defaultMessage="Settings"
            description="Label for the playground top-bar button that opens model parameters"
          />
        </Button>
      </Drawer.Trigger>
      <Drawer.Content
        componentId="mlflow.playground.params.drawer"
        title={intl.formatMessage({
          defaultMessage: 'Parameters',
          description: 'Title of the playground parameters drawer',
        })}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Leave a field blank to use the provider's default."
              description="Help text inside the playground parameters drawer"
            />
          </Typography.Hint>
          <ParametersForm value={value} onChange={onChange} />
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};
