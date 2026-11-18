import { Button, TagIcon, Tooltip } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { useLocation, useNavigate } from '../../../../../common/utils/RoutingUtils';
import { getPreservedQueryString } from '../../../../pages/experiment-page-tabs/side-nav/utils';
import Routes from '../../../../routes';
import { ExperimentPageTabName } from '../../../../constants';

export const EXPERIMENT_SETTINGS_TAGS_HASH = '#tags';

export interface ExperimentViewManageTagsButtonProps {
  experimentId: string;
  componentId: string;
}

/**
 * Opens the experiment Settings page scrolled to the Tags row in the About section.
 */
export const ExperimentViewManageTagsButton = ({ experimentId, componentId }: ExperimentViewManageTagsButtonProps) => {
  const intl = useIntl();
  const navigate = useNavigate();
  // Keep the workflow type and workspace, like the experiment sidebar links do.
  const preservedQuery = getPreservedQueryString(useLocation().search) ?? '';
  const label = intl.formatMessage({
    defaultMessage: 'Manage experiment tags',
    description: 'Label for the experiment header button that opens Settings to edit experiment tags',
  });

  return (
    <Tooltip componentId={`${componentId}.tooltip`} content={label}>
      <Button
        componentId={componentId}
        type="tertiary"
        size="small"
        icon={<TagIcon />}
        aria-label={label}
        onClick={() =>
          navigate(
            `${Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Settings)}${preservedQuery}${EXPERIMENT_SETTINGS_TAGS_HASH}`,
          )
        }
      />
    </Tooltip>
  );
};
