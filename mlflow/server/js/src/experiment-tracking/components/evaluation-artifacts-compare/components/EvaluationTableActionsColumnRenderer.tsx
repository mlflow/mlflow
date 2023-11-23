import { Button, ChevronDownIcon, ChevronRightIcon, Tooltip } from '@databricks/design-system';
import { EvaluationTableHeader } from './EvaluationTableHeader';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';
import { FormattedMessage } from 'react-intl';

const enlargedIconStyle = { svg: { width: 20, height: 20 } };

export const EvaluationTableActionsColumnRenderer = () => {
  const { toggleExpandedHeader, isHeaderExpanded } = usePromptEngineeringContext();

  return (
    <EvaluationTableHeader>
      <Tooltip
        placement='right'
        title={
          <FormattedMessage
            defaultMessage='Toggle detailed view'
            description='Experiment page > artifact compare view > table header > label for "toggle detailed view" button'
          />
        }
      >
        <Button
          icon={
            isHeaderExpanded ? (
              <ChevronDownIcon css={enlargedIconStyle} />
            ) : (
              <ChevronRightIcon css={enlargedIconStyle} />
            )
          }
          onClick={toggleExpandedHeader}
        />
      </Tooltip>
    </EvaluationTableHeader>
  );
};
