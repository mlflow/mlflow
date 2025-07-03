import React from 'react';
import { FormattedMessage } from 'react-intl';
import { ExperimentRunSearchSyntaxDocUrl } from '../../../../../common/constants';

export const RunsSearchTooltipContent = () => {
  return (
    <div className="search-input-tooltip-content">
      <FormattedMessage
        defaultMessage="Search runs using a simplified version of the SQL {whereBold} clause."
        description="Tooltip string to explain how to search runs from the experiments table"
        values={{ whereBold: <b>WHERE</b> }}
      />{' '}
      <FormattedMessage
        defaultMessage="<link>Learn more</link>"
        description="Learn more tooltip link to learn more on how to search in an experiments run table"
        values={{
          link: (chunks: any) => (
            <a href={ExperimentRunSearchSyntaxDocUrl} target="_blank" rel="noopener noreferrer">
              {chunks}
            </a>
          ),
        }}
      />
      <br />
      <FormattedMessage defaultMessage="Examples:" description="Text header for examples of mlflow search syntax" />
      <br />
      {'• metrics.rmse >= 0.8'}
      <br />
      {'• metrics.`f1 score` < 1'}
      <br />
      • params.model = 'tree'
      <br />
      • attributes.run_name = 'my run'
      <br />
      • tags.`mlflow.user` = 'myUser'
      <br />
      {"• metric.f1_score > 0.9 AND params.model = 'tree'"}
      <br />
      • dataset.name IN ('dataset1', 'dataset2')
      <br />
      • attributes.run_id IN ('a1b2c3d4', 'e5f6g7h8')
      <br />• tags.model_class LIKE 'sklearn.linear_model%'
    </div>
  );
};
