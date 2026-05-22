import type { ReactNode } from 'react';
import { Breadcrumb, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';

interface DatasetsBreadcrumbsProps {
  experimentId: string;
  /** Dataset name for the trailing crumb. Omit (or pass undefined) to show only "Datasets". */
  datasetName?: string;
  /**
   * Optional right-aligned slot rendered on the same row as the breadcrumb. Used on the
   * detail page to host the dataset-level kebab so it sits inline with the breadcrumb
   * without pushing the toolbar (and its search bar) below the list page's search bar.
   */
  rightActions?: ReactNode;
}

export const DatasetsBreadcrumbs = ({ experimentId, datasetName, rightActions }: DatasetsBreadcrumbsProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const listRoute = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets);
  const showTrailingCrumb = datasetName !== undefined;

  // Du Bois Breadcrumb renders as a <div>, not a <nav>. Wrapping in a labelled <nav> gives
  // the breadcrumb the WAI-ARIA navigation landmark expected of breadcrumbs and surfaces
  // it in screen-reader landmark menus alongside main/content navigation.
  //
  // Per-child `alignSelf` (instead of container `alignItems: center`) pins the breadcrumb
  // text to the top of the row while keeping `rightActions` (the kebab, when present) at the
  // row's vertical centre. Without this, the taller kebab pulls the centre-aligned text
  // down on the detail page so it no longer lines up with the list-page breadcrumb.
  return (
    <nav
      aria-label={intl.formatMessage({
        defaultMessage: 'Breadcrumb',
        description: 'Aria label for the V2 evaluation datasets breadcrumb navigation landmark',
      })}
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        // Match the default-size DS button height used by the detail-page kebab so the
        // breadcrumb row keeps the same height whether or not `rightActions` is rendered.
        // Without this, the toolbar (and its search bar) lands at a different Y on the
        // list page than on the detail page.
        minHeight: 32,
      }}
    >
      <div css={{ flex: 1, minWidth: 0, alignSelf: 'flex-start' }}>
        <Breadcrumb includeTrailingCaret={false}>
          <Breadcrumb.Item>
            {showTrailingCrumb ? (
              <Link to={listRoute}>
                <FormattedMessage
                  defaultMessage="Datasets"
                  description="Breadcrumb label for the V2 evaluation datasets list page"
                />
              </Link>
            ) : (
              // `aria-current="page"` marks the current location when no dataset is selected.
              <span aria-current="page">
                <FormattedMessage
                  defaultMessage="Datasets"
                  description="Breadcrumb label for the V2 evaluation datasets list page"
                />
              </span>
            )}
          </Breadcrumb.Item>
          {showTrailingCrumb && (
            <Breadcrumb.Item>
              <span aria-current="page">{datasetName}</span>
            </Breadcrumb.Item>
          )}
        </Breadcrumb>
      </div>
      {rightActions && <div css={{ flexShrink: 0, alignSelf: 'center' }}>{rightActions}</div>}
    </nav>
  );
};
