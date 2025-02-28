import { Overflow, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import { ReactComponent as RegisteredModelOkIcon } from '../../../../common/static/registered-model-grey-ok.svg';
import type { RunPageModelVersionSummary } from '../hooks/useUnifiedRegisteredModelVersionsSummariesForRun';

/**
 * Displays list of registered models in run detail overview.
 * TODO: expand with logged models after finalizing design
 */
export const RunViewRegisteredModelsBox = ({
  registeredModelVersionSummaries,
}: {
  registeredModelVersionSummaries: RunPageModelVersionSummary[];
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Overflow>
      {registeredModelVersionSummaries?.map((modelSummary) => (
        <Link
          key={modelSummary.displayedName}
          to={modelSummary.link}
          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
        >
          <RegisteredModelOkIcon /> {modelSummary.displayedName}{' '}
          <Tag
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewregisteredmodelsbox.tsx_40"
            css={{ cursor: 'pointer' }}
          >
            v{modelSummary.version}
          </Tag>
        </Link>
      ))}
    </Overflow>
  );
};
