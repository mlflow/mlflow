import { Button, ChevronDownIcon, ChevronRightIcon, LegacyTooltip } from '@databricks/design-system';
import { EvaluationTableHeader } from './EvaluationTableHeader';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';
import { FormattedMessage } from 'react-intl';

const enlargedIconStyle = { svg: { width: 20, height: 20 } };

export const EvaluationTableActionsColumnRenderer = () => {
  const { toggleExpandedHeader, isHeaderExpanded } = usePromptEngineeringContext();

  return (
    <EvaluationTableHeader>
      <LegacyTooltip
        placement="right"
        title={
          <FormattedMessage
            defaultMessage="Toggle detailed view"
            description='Experiment page > artifact compare view > table header > label for "toggle detailed view" button'
          />
        }
      >
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationtableactionscolumnrenderer.tsx_22"
          icon={
            isHeaderExpanded ? (
              <ChevronDownIcon css={enlargedIconStyle} />
            ) : (
              <ChevronRightIcon css={enlargedIconStyle} />
            )
          }
          onClick={toggleExpandedHeader}
        />
      </LegacyTooltip>
    </EvaluationTableHeader>
  );
};
