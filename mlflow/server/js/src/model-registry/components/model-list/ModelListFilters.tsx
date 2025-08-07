import {
  LegacyTooltip,
  TableFilterLayout,
  Button,
  TableFilterInput,
  InfoSmallIcon,
  Popover,
  Typography,
} from '@databricks/design-system';
import { useEffect, useState } from 'react';
import { FormattedMessage, defineMessage, useIntl } from 'react-intl';
import { ExperimentRunSearchSyntaxDocUrl } from '../../../common/constants';

export interface ModelListFiltersProps {
  searchFilter: string;
  onSearchFilterChange: (newValue: string) => void;
  isFiltered: boolean;
}

export const ModelSearchInputHelpTooltip = ({
  exampleEntityName = 'my_model_name',
}: {
  exampleEntityName?: string;
}) => {
  const { formatMessage } = useIntl();
  const tooltipIntroMessage = defineMessage({
    defaultMessage:
      'To search by tags or by names and tags, use a simplified version{newline}of the SQL {whereBold} clause.',
    description: 'Tooltip string to explain how to search models from the model registry table',
  });

  // Tooltips are not expected to contain links.
  const labelText = formatMessage(tooltipIntroMessage, { newline: ' ', whereBold: 'WHERE' });

  return (
    <Popover.Root componentId="codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_46">
      <Popover.Trigger
        aria-label={labelText}
        css={{ border: 0, background: 'none', padding: 0, lineHeight: 0, cursor: 'pointer' }}
      >
        <InfoSmallIcon />
      </Popover.Trigger>
      <Popover.Content align="start">
        <div>
          <FormattedMessage {...tooltipIntroMessage} values={{ newline: <br />, whereBold: <b>WHERE</b> }} />{' '}
          <FormattedMessage
            defaultMessage="<link>Learn more</link>"
            description="Learn more tooltip link to learn more on how to search models"
            values={{
              link: (chunks) => (
                <Typography.Link
                  componentId="codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_61"
                  href={ExperimentRunSearchSyntaxDocUrl + '#syntax'}
                  openInNewTab
                >
                  {chunks}
                </Typography.Link>
              ),
            }}
          />
          <br />
          <br />
          <FormattedMessage defaultMessage="Examples:" description="Text header for examples of mlflow search syntax" />
          <br />
          • tags.my_key = "my_value"
          <br />• name ilike "%{exampleEntityName}%" and tags.my_key = "my_value"
        </div>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};

export const ModelListFilters = ({
  // prettier-ignore
  searchFilter,
  onSearchFilterChange,
  isFiltered,
}: ModelListFiltersProps) => {
  const intl = useIntl();

  const [internalSearchFilter, setInternalSearchFilter] = useState(searchFilter);

  const triggerSearch = () => {
    onSearchFilterChange(internalSearchFilter);
  };
  useEffect(() => {
    setInternalSearchFilter(searchFilter);
  }, [searchFilter]);

  const reset = () => {
    onSearchFilterChange('');
  };

  return (
    <TableFilterLayout>
      <TableFilterInput
        componentId="codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_118"
        placeholder={intl.formatMessage({
          defaultMessage: 'Filter registered models by name or tags',
          description: 'Placeholder text inside model search bar',
        })}
        onSubmit={triggerSearch}
        onClear={() => {
          setInternalSearchFilter('');
          onSearchFilterChange('');
        }}
        onChange={(e) => setInternalSearchFilter(e.target.value)}
        data-testid="model-search-input"
        suffix={<ModelSearchInputHelpTooltip />}
        value={internalSearchFilter}
        showSearchButton
      />
      {isFiltered && (
        <Button
          componentId="codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_152"
          type="tertiary"
          onClick={reset}
          data-testid="models-list-filters-reset"
        >
          <FormattedMessage defaultMessage="Reset filters" description="Reset filters button in list" />
        </Button>
      )}
    </TableFilterLayout>
  );
};
