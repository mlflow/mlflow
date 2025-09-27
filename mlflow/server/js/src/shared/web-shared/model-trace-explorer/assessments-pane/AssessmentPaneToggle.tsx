import { GavelIcon, SegmentedControlGroup, SegmentedControlButton } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const AssessmentPaneToggle = ({ allowInComparisonView = false }: { allowInComparisonView?: boolean }) => {
  const { assessmentsPaneExpanded, setAssessmentsPaneExpanded, assessmentsPaneEnabled, isInComparisonView } =
    useModelTraceExplorerViewState();

  if (isInComparisonView && !allowInComparisonView) {
    return null;
  }

  return (
    <SegmentedControlGroup
      css={{ display: 'block' }}
      name="shared.model-trace-explorer.assessments-pane-toggle"
      componentId="shared.model-trace-explorer.assessments-pane-toggle"
      value={assessmentsPaneExpanded}
      size="small"
    >
      <SegmentedControlButton
        value
        disabled={!assessmentsPaneEnabled}
        icon={<GavelIcon />}
        onClick={() => setAssessmentsPaneExpanded?.(!assessmentsPaneExpanded)}
        css={{
          '& > span': {
            display: 'flex',
            alignItems: 'center',
          },
        }}
      >
        {(!isInComparisonView || allowInComparisonView) && assessmentsPaneExpanded && (
          <FormattedMessage
            defaultMessage="Assessments"
            description="Label for the assessments pane of the model trace explorer."
          />
        )}
      </SegmentedControlButton>
    </SegmentedControlGroup>
  );
};
