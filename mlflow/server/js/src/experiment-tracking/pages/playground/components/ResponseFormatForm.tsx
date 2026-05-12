import {
  FormUI,
  Input,
  SegmentedControlButton,
  SegmentedControlGroup,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ResponseFormatType } from '../types';

const { TextArea } = Input;

interface Props {
  type: ResponseFormatType;
  onTypeChange: (next: ResponseFormatType) => void;
  schemaText: string;
  onSchemaChange: (next: string) => void;
  schemaError?: string | null;
}

const TYPE_OPTIONS: { value: ResponseFormatType; label: string }[] = [
  { value: 'text', label: 'Text' },
  { value: 'json_object', label: 'JSON' },
  { value: 'json_schema', label: 'JSON schema' },
];

const SCHEMA_PLACEHOLDER = `{
  "name": "weather_report",
  "schema": {
    "type": "object",
    "properties": {
      "location": { "type": "string" },
      "temperature": { "type": "number" }
    },
    "required": ["location", "temperature"]
  },
  "strict": true
}`;

export const ResponseFormatForm = ({ type, onTypeChange, schemaText, onSchemaChange, schemaError }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <SegmentedControlGroup
        componentId="mlflow.playground.response_format.type"
        name="mlflow.playground.response_format.type"
        size="small"
        value={type}
        onChange={(event) => onTypeChange(event.target.value as ResponseFormatType)}
      >
        {TYPE_OPTIONS.map(({ value, label }) => (
          <SegmentedControlButton key={value} value={value}>
            {label}
          </SegmentedControlButton>
        ))}
      </SegmentedControlGroup>
      {type === 'json_schema' && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <FormUI.Label htmlFor="mlflow.playground.response_format.schema">
            <FormattedMessage
              defaultMessage="Schema"
              description="Label for the playground response_format JSON schema textarea"
            />
          </FormUI.Label>
          <TextArea
            componentId="mlflow.playground.response_format.schema"
            id="mlflow.playground.response_format.schema"
            value={schemaText}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onSchemaChange(event.target.value)}
            autoSize={{ minRows: 4, maxRows: 16 }}
            placeholder={intl.formatMessage(
              {
                defaultMessage: 'e.g. {example}',
                description: 'Placeholder shown inside the playground response_format JSON schema textarea',
              },
              { example: SCHEMA_PLACEHOLDER },
            )}
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
            }}
          />
          {schemaError && <FormUI.Message type="error" message={schemaError} />}
        </div>
      )}
    </div>
  );
};
