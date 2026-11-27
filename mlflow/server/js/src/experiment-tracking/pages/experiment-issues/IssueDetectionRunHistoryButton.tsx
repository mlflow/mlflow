import { Button, HistoryIcon, Tooltip } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

interface IssueDetectionRunHistoryButtonProps {
  experimentId: string;
  componentId: string;
}

export const IssueDetectionRunHistoryButton = ({ experimentId, componentId }: IssueDetectionRunHistoryButtonProps) => {
  const intl = useIntl();
  const navigate = useNavigate();
  const label = intl.formatMessage({
    defaultMessage: 'Issue detection run history',
    description: 'Accessible label for the button opening issue detection run history',
  });

  return (
    <Tooltip componentId={`${componentId}.tooltip`} content={label}>
      <Button
        componentId={componentId}
        icon={<HistoryIcon />}
        aria-label={label}
        onClick={() => navigate(Routes.getIssueDetectionRunsRoute(experimentId))}
      />
    </Tooltip>
  );
};
