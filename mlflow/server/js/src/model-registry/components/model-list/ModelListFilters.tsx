import {
  Popover,
  TableFilterLayout,
  Button,
  TableFilterInput,
  InfoIcon,
} from '@databricks/design-system';
import { useEffect, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { ExperimentSearchSyntaxDocUrl } from '../../../common/constants';

export interface ModelListFiltersProps {
  searchFilter: string;
  onSearchFilterChange: (newValue: string) => void;
  isFiltered: boolean;
}

const ModelSearchInputHelpTooltip = () => {
  return (
    <Popover
      content={
        <div>
          <FormattedMessage
            defaultMessage='To search by tags or by names and tags, use a simplified version{newline}of the SQL {whereBold} clause.'
            description='Tooltip string to explain how to search models from the model registry table'
            values={{ newline: <br />, whereBold: <b>WHERE</b> }}
          />{' '}
          <FormattedMessage
            defaultMessage='<link>Learn more</link>'
            description='Learn more tooltip link to learn more on how to search models'
            values={{
              link: (chunks) => (
                <a
                  href={ExperimentSearchSyntaxDocUrl + '#syntax'}
                  target='_blank'
                  rel='noopener noreferrer'
                >
                  {chunks}
                </a>
              ),
            }}
          />
          <br />
          <FormattedMessage
            defaultMessage='Examples:'
            description='Text header for examples of mlflow search syntax'
          />
          <br />
          {'• tags.my_key = "my_value"'}
          <br />
          {'• name ilike "%my_model_name%" and tags.my_key = "my_value"'}
        </div>
      }
      placement='bottom'
    >
      <InfoIcon css={{ cursor: 'pointer' }} />
    </Popover>
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
        placeholder={intl.formatMessage({
          defaultMessage: 'Filter models',
          description: 'Placeholder text inside model search bar',
        })}
        onSubmit={triggerSearch}
        onChange={(e) => setInternalSearchFilter(e.target.value)}
        data-testid='model-search-input'
        allowClear={false}
        suffix={<ModelSearchInputHelpTooltip />}
        value={internalSearchFilter}
        showSearchButton
      />
      {isFiltered && (
        <Button type='link' onClick={reset} data-testid='models-list-filters-reset'>
          <FormattedMessage
            defaultMessage='Reset filters'
            description='Models table > filters > reset filters button'
          />
        </Button>
      )}
    </TableFilterLayout>
  );
};
