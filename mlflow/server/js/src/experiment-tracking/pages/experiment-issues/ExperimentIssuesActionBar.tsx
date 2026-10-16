import { Input, SearchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { DetectIssuesButton } from '@databricks/web-shared/genai-traces-table/components/DetectIssuesButton';
import { ExperimentIssuesFilterControl } from './ExperimentIssuesToolbar';
import type { ExperimentIssuesToolbarControls } from './useExperimentIssuesListControls';

const COMPONENT_ID = 'mlflow.experiment-issues.action-bar';

export interface ExperimentIssuesActionBarProps {
  controls: ExperimentIssuesToolbarControls;
  /** The search box's text as typed, which may not have been committed to the search yet. */
  searchInput: string;
  onSearchInputChange: (value: string) => void;
  /** Commits the typed search now, rather than after the debounce. */
  onSearchSubmit: () => void;
  onSearchClear: () => void;
  /** Loading state comes from the OSS job lifecycle. */
  isDetecting: boolean;
  detectionEnabled: boolean;

  onRunDetection: () => void;
}

/**
 * The row above the issue list and detail: the title search and filters that narrow
 * the list, then the actions that produce new Issues. The active filter chips sit with the list.
 */
export const ExperimentIssuesActionBar = ({
  controls,
  searchInput,
  onSearchInputChange,
  onSearchSubmit,
  onSearchClear,
  isDetecting,
  detectionEnabled,
  onRunDetection,
}: ExperimentIssuesActionBarProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div
      role="toolbar"
      aria-label={intl.formatMessage({
        defaultMessage: 'Search and detect issues',
        description:
          'Experiment issues tab > Accessible label of the row with the search, filters and detection actions',
      })}
      css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
    >
      {/* Sized by its wrapper, so the box shrinks before the actions on the right wrap. */}
      <div css={{ flex: '0 1 400px', minWidth: 0 }}>
        <Input
          componentId={`${COMPONENT_ID}.search`}
          prefix={<SearchIcon />}
          allowClear
          value={searchInput}
          onChange={(event) => onSearchInputChange(event.target.value)}
          onClear={onSearchClear}
          onPressEnter={onSearchSubmit}
          placeholder={intl.formatMessage({
            defaultMessage: 'Search issues by title',
            description: 'Experiment issues tab > Placeholder of the search box that matches issue titles',
          })}
          aria-label={intl.formatMessage({
            defaultMessage: 'Search issues by title',
            description: 'Experiment issues tab > Accessible label of the search box that matches issue titles',
          })}
        />
      </div>
      <ExperimentIssuesFilterControl controls={controls} align="start" />
      <div css={{ flex: 1 }} />
      {detectionEnabled && (
        <DetectIssuesButton
          componentId="mlflow.experiment-issues.detect"
          onClick={onRunDetection}
          loading={isDetecting}
        />
      )}
    </div>
  );
};
