import { GavelIcon, SegmentedControlGroup, SegmentedControlButton } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const AssessmentPaneToggle = () => {
  const { assessmentsPaneExpanded, setAssessmentsPaneExpanded, assessmentsPaneEnabled } =
    useModelTraceExplorerViewState();

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
        {!assessmentsPaneExpanded && (
          <FormattedMessage
            defaultMessage="Assessments"
            description="Label for the assessments pane of the model trace explorer."
          />
        )}
      </SegmentedControlButton>
    </SegmentedControlGroup>
  );
};
