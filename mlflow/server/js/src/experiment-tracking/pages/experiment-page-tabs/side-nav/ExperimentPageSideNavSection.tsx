import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import {
  COLLAPSED_CLASS_NAME,
  FULL_WIDTH_CLASS_NAME,
  getExperimentPageSideNavSectionLabel,
  type ExperimentPageSideNavItem,
  type ExperimentPageSideNavSectionKey,
} from './constants';
import { ExperimentPageTabName } from '../../../constants';
import { Link, useLocation, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import invariant from 'invariant';
import { isTracesRelatedTab } from './utils';
import { useLogTelemetryEvent } from '@mlflow/mlflow/src/telemetry/hooks/useLogTelemetryEvent';
import { useMemo } from 'react';

export const ExperimentPageSideNavSection = ({
  sectionKey,
  activeTab,
  items,
}: {
  sectionKey: ExperimentPageSideNavSectionKey;
  activeTab: ExperimentPageTabName;
  items: ExperimentPageSideNavItem[];
}) => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams();
  const { search } = useLocation();
  const logTelemetryEvent = useLogTelemetryEvent();
  const viewId = useMemo(() => crypto.randomUUID(), []);

  invariant(experimentId, 'Experiment ID must be defined');

  // NOTE: everything with `className={COLLAPSED_CLASS_NAME}` is hidden at
  // large screen sizes (browser's XL breakpoint). The`display` property is
  // controlled via media query in the parent component. This is why there are
  // seemingly duplicate elements in the code below.
  return (
    <div css={{ display: 'flex', flexDirection: 'column', marginBottom: theme.spacing.sm + theme.spacing.xs }}>
      {sectionKey !== 'top-level' && (
        <div
          css={{
            display: 'flex',
            marginBottom: theme.spacing.xs,
            position: 'relative',
            height: theme.typography.lineHeightBase,
          }}
        >
          <div
            className={COLLAPSED_CLASS_NAME}
            css={{
              border: 'none',
              borderBottom: `1px solid ${theme.colors.border}`,
              width: '100%',
              position: 'absolute',
              bottom: '50%',
            }}
          />
          <Typography.Text className={FULL_WIDTH_CLASS_NAME} size="sm" color="secondary">
            {getExperimentPageSideNavSectionLabel(sectionKey as ExperimentPageSideNavSectionKey)}
          </Typography.Text>
        </div>
      )}
      {items.map((item) => {
        // SingleChatSession is a special case because it's a nested tab
        const isActive =
          activeTab === ExperimentPageTabName.SingleChatSession
            ? item.tabName === ExperimentPageTabName.ChatSessions
            : activeTab === item.tabName;

        const preserveQueryParams = isTracesRelatedTab(activeTab) && isTracesRelatedTab(item.tabName);

        return (
          <Link
            key={`${sectionKey}-${item.tabName}`}
            to={{
              pathname: Routes.getExperimentPageTabRoute(experimentId, item.tabName),
              search: preserveQueryParams ? search : undefined,
            }}
            onClick={() =>
              logTelemetryEvent({
                componentId: item.componentId,
                componentViewId: viewId,
                componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
                componentSubType: null,
                eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
              })
            }
          >
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderRadius: theme.borders.borderRadiusSm,
                cursor: 'pointer',
                backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : undefined,
                color: isActive ? theme.colors.actionDefaultIconHover : theme.colors.actionDefaultIconDefault,
                height: theme.typography.lineHeightBase,
                boxSizing: 'content-box',
                ':hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
              }}
            >
              <Tooltip
                componentId={`mlflow.experiment-page.side-nav.${sectionKey}.${item.tabName}.tooltip`}
                content={item.label}
                side="right"
                delayDuration={0}
              >
                <span className={COLLAPSED_CLASS_NAME}>{item.icon}</span>
              </Tooltip>
              <span className={FULL_WIDTH_CLASS_NAME}>{item.icon}</span>
              <Typography.Text className={FULL_WIDTH_CLASS_NAME} bold={isActive} color="primary">
                {item.label}
              </Typography.Text>
            </div>
          </Link>
        );
      })}
    </div>
  );
};
