import { Button, SidebarCollapseIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const AssessmentPaneToggle = () => {
  const { assessmentsPaneExpanded, setAssessmentsPaneExpanded, assessmentsPaneEnabled } =
    useModelTraceExplorerViewState();

  if (assessmentsPaneExpanded) {
    return null;
  }

  return (
    <Button
      disabled={!assessmentsPaneEnabled}
      type="primary"
      componentId="shared.model-trace-explorer.assessments-pane-toggle"
      size="small"
      icon={<SidebarCollapseIcon />}
      onClick={() => setAssessmentsPaneExpanded?.(true)}
    >
      <FormattedMessage
        defaultMessage="Show assessments"
        description="Label for the button to show the assessments pane"
      />
    </Button>
  );
};
