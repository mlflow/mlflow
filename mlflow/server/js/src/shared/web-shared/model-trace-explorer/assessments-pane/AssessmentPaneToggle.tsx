import { useDesignSystemTheme, SidebarExpandIcon, Button, SidebarCollapseIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const AssessmentPaneToggle = () => {
  const { assessmentsPaneExpanded, setAssessmentsPaneExpanded, assessmentsPaneEnabled, isInComparisonView } =
    useModelTraceExplorerViewState();

  if (isInComparisonView) {
    return null;
  }

  return (
    <Button
      disabled={!assessmentsPaneEnabled}
      type="primary"
      componentId="shared.model-trace-explorer.assessments-pane-toggle"
      size="small"
      icon={assessmentsPaneExpanded ? <SidebarExpandIcon /> : <SidebarCollapseIcon />}
      onClick={() => setAssessmentsPaneExpanded?.(!assessmentsPaneExpanded)}
    >
      {assessmentsPaneExpanded ? (
        <FormattedMessage
          defaultMessage="Hide assessments"
          description="Label for the button to hide the assessments pane"
        />
      ) : (
        <FormattedMessage
          defaultMessage="Show assessments"
          description="Label for the button to show the assessments pane"
        />
      )}
    </Button>
  );
};
