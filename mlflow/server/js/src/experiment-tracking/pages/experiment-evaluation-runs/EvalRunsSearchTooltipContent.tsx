import { FormattedMessage } from 'react-intl';
import { ExperimentRunSearchSyntaxDocUrl } from '../../../common/constants';

/**
 * Info-tooltip body for the eval-runs search box. Same run-search syntax as the classic runs page, but the
 * examples are framed around GenAI evaluation (quality scorers, model/prompt versions, eval datasets) so the
 * guidance reads naturally for this surface.
 */
export const EvalRunsSearchTooltipContent = () => {
  return (
    <div className="search-input-tooltip-content">
      <FormattedMessage
        defaultMessage="Search evaluation runs using a simplified version of the SQL {whereBold} clause."
        description="Tooltip string to explain how to search evaluation runs from the eval-runs table"
        values={{ whereBold: <b>WHERE</b> }}
      />{' '}
      <FormattedMessage
        defaultMessage="<link>Learn more</link>"
        description="Learn more tooltip link explaining how to search in the evaluation runs table"
        values={{
          link: (chunks) => (
            <a href={ExperimentRunSearchSyntaxDocUrl} target="_blank" rel="noopener noreferrer">
              {chunks}
            </a>
          ),
        }}
      />
      <br />
      <FormattedMessage
        defaultMessage="Examples:"
        description="Text header for examples of the evaluation runs search syntax"
      />
      <br />
      {'• metrics.correctness >= 0.8'}
      <br />
      {'• metrics.`answer relevance` < 0.5'}
      <br />• params.model = 'gpt-4o'
      <br />• params.prompt_version = 'v3'
      <br />• attributes.run_name = 'nightly-eval'
      <br />
      {"• metrics.groundedness > 0.9 AND params.model = 'gpt-4o'"}
      <br />• dataset.name IN ('golden-set', 'regression-set')
    </div>
  );
};
