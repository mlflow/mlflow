import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useEndpointsQuery } from '../../../../gateway/hooks/useEndpointsQuery';

const COMPONENT_ID = 'mlflow.playground.endpoint_picker';

interface Props {
  value?: string;
  onChange: (endpointName: string) => void;
}

export const EndpointPicker = ({ value, onChange }: Props) => {
  const intl = useIntl();
  const { data: endpoints, isLoading, error } = useEndpointsQuery();

  return (
    <div>
      <FormUI.Label htmlFor={COMPONENT_ID}>
        <FormattedMessage
          defaultMessage="Endpoint"
          description="Label for the AI gateway endpoint picker on the playground page"
        />
      </FormUI.Label>
      <DialogCombobox
        componentId={COMPONENT_ID}
        label={intl.formatMessage({
          defaultMessage: 'Endpoint',
          description: 'Label for the AI gateway endpoint picker on the playground page',
        })}
        modal={false}
        value={value ? [value] : undefined}
      >
        <DialogComboboxTrigger
          id={COMPONENT_ID}
          css={{ width: '100%' }}
          allowClear
          placeholder={intl.formatMessage({
            defaultMessage: 'Select a gateway endpoint',
            description: 'Placeholder for the AI gateway endpoint picker on the playground page',
          })}
          withInlineLabel={false}
          onClear={() => onChange('')}
        />
        <DialogComboboxContent loading={isLoading} maxHeight={400} matchTriggerWidth>
          {!isLoading && (
            <DialogComboboxOptionList>
              <DialogComboboxOptionListSearch autoFocus>
                {endpoints.map((endpoint) => (
                  <DialogComboboxOptionListSelectItem
                    value={endpoint.name}
                    key={endpoint.endpoint_id}
                    onChange={(name) => onChange(name)}
                    checked={value === endpoint.name}
                  >
                    {endpoint.name}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionListSearch>
            </DialogComboboxOptionList>
          )}
        </DialogComboboxContent>
      </DialogCombobox>
      {error && <FormUI.Message type="error" message={error.message} />}
    </div>
  );
};
