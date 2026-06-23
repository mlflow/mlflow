import {
  Button,
  FormUI,
  IndentIncreaseIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ResponseFormatType } from '../types';
import { formatJson } from '../utils';
import { LazyJsonRecordEditor } from '../../experiment-evaluation-datasets-v2/components/LazyJsonRecordEditor';

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
          <div
            css={{
              display: 'flex',
              alignItems: 'flex-end',
              justifyContent: 'space-between',
              marginBottom: theme.spacing.sm,
              '& label': { marginBottom: 0 },
            }}
          >
            <FormUI.Label htmlFor="mlflow.playground.response_format.schema">
              <FormattedMessage
                defaultMessage="Schema"
                description="Label for the playground response_format JSON schema editor"
              />
            </FormUI.Label>
            <Button
              componentId="mlflow.playground.response_format.format"
              size="small"
              icon={<IndentIncreaseIcon />}
              disabled={Boolean(schemaError)}
              onClick={() => {
                const formatted = formatJson(schemaText);
                if (formatted !== null) {
                  onSchemaChange(formatted);
                }
              }}
            >
              <FormattedMessage
                defaultMessage="Format"
                description="Button that pretty-prints the response format JSON schema in the playground"
              />
            </Button>
          </div>
          <LazyJsonRecordEditor
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Schema',
              description: 'Accessible label for the playground response_format JSON schema editor',
            })}
            value={schemaText}
            onChange={onSchemaChange}
            height="160px"
            maxHeight="360px"
            transparentBackground
            errorMessage={schemaError ?? undefined}
          />
        </div>
      )}
    </div>
  );
};
