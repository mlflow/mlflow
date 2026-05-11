import {
  ChevronDownIcon,
  ChevronRightIcon,
  FormUI,
  Input,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { PlaygroundParams } from '../types';

const { TextArea } = Input;

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

const stopToText = (stop: string[] | undefined): string => (stop ?? []).join('\n');

const stopFromText = (text: string): string[] | undefined => {
  const sequences = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  return sequences.length > 0 ? sequences : undefined;
};

export const ParametersForm = ({ value, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleNumber =
    (field: 'temperature' | 'max_tokens' | 'top_p' | 'top_k' | 'presence_penalty' | 'frequency_penalty') =>
    (event: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...value, [field]: parseNumber(event.target.value) });
    };

  const handleStop = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange({ ...value, stop: stopFromText(event.target.value) });
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        padding: theme.spacing.md,
        '& label': { fontWeight: theme.typography.typographyRegularFontWeight },
      }}
    >
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
        onChange={handleNumber('temperature')}
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
        onChange={handleNumber('max_tokens')}
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
        onChange={handleNumber('top_p')}
      />

      <button
        type="button"
        onClick={() => setShowAdvanced((v) => !v)}
        aria-expanded={showAdvanced}
        css={{
          alignSelf: 'flex-start',
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
        {showAdvanced ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <FormattedMessage
          defaultMessage="Advanced"
          description="Toggle label that reveals advanced sampling parameters on the playground"
        />
      </button>

      {showAdvanced && (
        <>
          <FormUI.Label htmlFor="mlflow.playground.params.top_k">
            <FormattedMessage
              defaultMessage="Top K"
              description="Label for the top-k input on the playground parameters panel"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.playground.params.top_k"
            id="mlflow.playground.params.top_k"
            type="number"
            min={1}
            step={1}
            value={formatValue(value.top_k)}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g. 40',
              description: 'Placeholder for the top-k input on the playground parameters panel',
            })}
            onChange={handleNumber('top_k')}
          />

          <FormUI.Label htmlFor="mlflow.playground.params.presence_penalty">
            <FormattedMessage
              defaultMessage="Presence penalty"
              description="Label for the presence-penalty input on the playground parameters panel"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.playground.params.presence_penalty"
            id="mlflow.playground.params.presence_penalty"
            type="number"
            min={-2}
            max={2}
            step={0.1}
            value={formatValue(value.presence_penalty)}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g. 0.0',
              description: 'Placeholder for the presence-penalty input on the playground parameters panel',
            })}
            onChange={handleNumber('presence_penalty')}
          />

          <FormUI.Label htmlFor="mlflow.playground.params.frequency_penalty">
            <FormattedMessage
              defaultMessage="Frequency penalty"
              description="Label for the frequency-penalty input on the playground parameters panel"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.playground.params.frequency_penalty"
            id="mlflow.playground.params.frequency_penalty"
            type="number"
            min={-2}
            max={2}
            step={0.1}
            value={formatValue(value.frequency_penalty)}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g. 0.0',
              description: 'Placeholder for the frequency-penalty input on the playground parameters panel',
            })}
            onChange={handleNumber('frequency_penalty')}
          />

          <FormUI.Label htmlFor="mlflow.playground.params.stop">
            <FormattedMessage
              defaultMessage="Stop sequences"
              description="Label for the stop-sequences input on the playground parameters panel"
            />
          </FormUI.Label>
          <TextArea
            componentId="mlflow.playground.params.stop"
            id="mlflow.playground.params.stop"
            value={stopToText(value.stop)}
            onChange={handleStop}
            autoSize={{ minRows: 1, maxRows: 6 }}
            placeholder={intl.formatMessage({
              defaultMessage: 'One sequence per line',
              description: 'Placeholder for the stop-sequences textarea on the playground parameters panel',
            })}
          />
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="The model stops generating when it produces any of these strings."
              description="Help text under the stop-sequences textarea on the playground parameters panel"
            />
          </Typography.Hint>
        </>
      )}
    </div>
  );
};
