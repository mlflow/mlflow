import { Button, Empty, NoIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';

export const ExperimentViewNoPermissionsError = () => {
  return (
    <div css={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        description={
          <FormattedMessage
            defaultMessage="You don't have permissions to open requested experiment."
            description="A message shown on the experiment page if user has no permissions to open the experiment"
          />
        }
        image={<NoIcon />}
        title={
          <FormattedMessage
            defaultMessage="Permission denied"
            description="A title shown on the experiment page if user has no permissions to open the experiment"
          />
        }
      />
    </div>
  );
};
