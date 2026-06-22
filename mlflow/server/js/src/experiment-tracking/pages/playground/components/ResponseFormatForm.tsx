import { FormUI, SegmentedControlButton, SegmentedControlGroup, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ResponseFormatType } from '../types';
import { JsonEditor } from './JsonEditor';

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
  "type": "object",
  "properties": {
    "location": { "type": "string" },
    "temperature": { "type": "number" }
  },
  "required": ["location", "temperature"]
}`;

export const ResponseFormatForm = ({ type, onTypeChange, schemaText, onSchemaChange, schemaError }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <SegmentedControlGroup
        componentId="mlflow.playground.response_format.type"
        name="mlflow.playground.response_format.type"
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
        <div>
          <FormUI.Label htmlFor="mlflow.playground.response_format.schema">
            <FormattedMessage
              defaultMessage="Schema"
              description="Label for the playground response_format JSON schema editor"
            />
          </FormUI.Label>
          <JsonEditor
            id="mlflow.playground.response_format.schema"
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Schema',
              description: 'Accessible label for the playground response_format JSON schema editor',
            })}
            value={schemaText}
            onChange={onSchemaChange}
            placeholder={SCHEMA_PLACEHOLDER}
            invalid={Boolean(schemaError)}
          />
          {schemaError && <FormUI.Message type="error" message={schemaError} />}
        </div>
      )}
    </div>
  );
};
