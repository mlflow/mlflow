import { Button, Empty, NoIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';

export const ExperimentViewNotFound = () => {
  return (
    <div css={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        description={
          <FormattedMessage
            defaultMessage="Requested experiment was not found."
            description="A message shown on the experiment page if the experiment is not found"
          />
        }
        image={<NoIcon />}
        title={
          <FormattedMessage
            defaultMessage="Experiment not found"
            description="A title shown on the experiment page if the experiment is not found"
          />
        }
      />
    </div>
  );
};
