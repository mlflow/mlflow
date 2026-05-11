import {
  FormUI,
  Input,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
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

const TYPE_OPTIONS: ResponseFormatType[] = ['text', 'json_object', 'json_schema'];

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
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <SegmentedControlGroup
        componentId="mlflow.playground.response_format.type"
        name="mlflow.playground.response_format.type"
        size="small"
        value={type}
        onChange={(event) => onTypeChange(event.target.value as ResponseFormatType)}
      >
        {TYPE_OPTIONS.map((option) => (
          <SegmentedControlButton key={option} value={option}>
            {option}
          </SegmentedControlButton>
        ))}
      </SegmentedControlGroup>
      <Typography.Hint>
        <FormattedMessage
          defaultMessage="text — model's free response. json_object — model returns valid JSON. json_schema — model output conforms to the schema you provide."
          description="Help text under the playground response_format picker"
        />
      </Typography.Hint>
      {type === 'json_schema' && (
        <>
          <FormUI.Label htmlFor="mlflow.playground.response_format.schema">
            <FormattedMessage
              defaultMessage="JSON schema"
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
        </>
      )}
    </div>
  );
};
