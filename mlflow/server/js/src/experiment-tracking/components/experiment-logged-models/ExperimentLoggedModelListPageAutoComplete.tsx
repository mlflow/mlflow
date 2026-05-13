import { FormattedMessage } from 'react-intl';
import { EntitySearchAutoComplete } from '../EntitySearchAutoComplete';
import type { LoggedModelProto } from '../../types';
import { useMemo } from 'react';
import type { EntitySearchAutoCompleteEntityNameGroup } from '../EntitySearchAutoComplete.utils';
import { getEntitySearchOptionsFromEntityNames } from '../EntitySearchAutoComplete.utils';
import { isUserFacingTag } from '../../../common/utils/TagUtils';

const getEntityNamesFromLoggedModelsData = (
  loggedModels: LoggedModelProto[],
): EntitySearchAutoCompleteEntityNameGroup => {
  const metricNames = new Set<string>();
  const paramNames = new Set<string>();
  const tagNames = new Set<string>();

  for (const loggedModel of loggedModels) {
    loggedModel.data?.metrics?.forEach((metric) => metric.key && metricNames.add(metric.key));
    loggedModel.data?.params?.forEach((param) => param.key && paramNames.add(param.key));
    loggedModel.info?.tags?.forEach((tag) => tag.key && tagNames.add(tag.key));
  }

  return {
    metricNames: Array.from(metricNames),
    paramNames: Array.from(paramNames),
    tagNames: Array.from(tagNames).filter(isUserFacingTag),
  };
};

const VALID_FILTER_ATTRIBUTES = [
  'model_id',
  'model_name',
  'status',
  'artifact_uri',
  'creation_time',
  'last_updated_time',
];

export const ExperimentLoggedModelListPageAutoComplete = ({
  searchQuery,
  onChangeSearchQuery,
  loggedModelsData,
}: {
  searchQuery?: string;
  onChangeSearchQuery: (searchFilter: string) => void;
  loggedModelsData: LoggedModelProto[];
}) => {
  const options = useMemo(() => {
    const entityNames = getEntityNamesFromLoggedModelsData(loggedModelsData);
    const validAttributeOptions = VALID_FILTER_ATTRIBUTES.map((attribute) => ({
      value: `attributes.${attribute}`,
    }));
    return getEntitySearchOptionsFromEntityNames(entityNames, validAttributeOptions);
  }, [loggedModelsData]);

  return (
    <EntitySearchAutoComplete
      searchFilter={searchQuery ?? ''}
      onSearchFilterChange={onChangeSearchQuery}
      defaultActiveFirstOption={false}
      baseOptions={options}
      onClear={() => onChangeSearchQuery('')}
      placeholder="metrics.rmse >= 0.8"
      tooltipContent={
        <div>
          <FormattedMessage
            defaultMessage="Search logged models using a simplified version of the SQL {whereBold} clause."
            description="Tooltip string to explain how to search logged models from the listing page"
            values={{ whereBold: <b>WHERE</b> }}
          />{' '}
          <br />
          <FormattedMessage
            defaultMessage="Examples:"
            description="Text header for examples of logged models search syntax"
          />
          <br />
          {'• metrics.rmse >= 0.8'}
          <br />
          {'• metrics.`f1 score` < 1'}
          <br />
          • params.type = 'tree'
          <br />
          • tags.my_tag = 'foobar'
          <br />
          • attributes.name = 'elasticnet'
          <br />
        </div>
      }
    />
  );
};
