import { SidebarExpandIcon, Button, SidebarCollapseIcon, Tooltip } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const AssessmentPaneToggle = () => {
  const { assessmentsPaneExpanded, setAssessmentsPaneExpanded, assessmentsPaneEnabled } =
    useModelTraceExplorerViewState();
  const intl = useIntl();

  const label = assessmentsPaneExpanded
    ? intl.formatMessage({
        defaultMessage: 'Hide assessments',
        description: 'Label for the button to hide the assessments pane',
      })
    : intl.formatMessage({
        defaultMessage: 'Show assessments',
        description: 'Label for the button to show the assessments pane',
      });

  if (assessmentsPaneExpanded) {
    return null;
  }

  return (
    <Tooltip componentId="shared.model-trace-explorer.assessments-pane-toggle-tooltip" content={label}>
      <Button
        disabled={!assessmentsPaneEnabled}
        type="primary"
        componentId="shared.model-trace-explorer.assessments-pane-toggle"
        size="small"
        icon={assessmentsPaneExpanded ? <SidebarExpandIcon /> : <SidebarCollapseIcon />}
        onClick={() => setAssessmentsPaneExpanded?.(!assessmentsPaneExpanded)}
        aria-label={label}
      />
    </Tooltip>
  );
};
