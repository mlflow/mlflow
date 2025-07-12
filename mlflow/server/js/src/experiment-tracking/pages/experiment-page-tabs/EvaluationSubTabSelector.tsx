import { Spacer, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentPageTabName } from '../../constants';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { Link } from '../../../common/utils/RoutingUtils';
import { FormattedMessage } from 'react-intl';

export const EvaluationSubTabSelector = ({
  experimentId,
  activeTab,
}: {
  experimentId: string;
  activeTab: ExperimentPageTabName;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <>
      <Spacer size="sm" shrinks={false} />
      <div css={{ width: '100%', borderTop: `1px solid ${theme.colors.border}` }} />
    </>
  );
};
