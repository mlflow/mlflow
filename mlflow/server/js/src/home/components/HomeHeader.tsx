import { Alert, Button, Header, Spacer } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

type HomeHeaderProps = {
  onCreateExperiment: () => void;
};

export const HomeHeader = ({ onCreateExperiment }: HomeHeaderProps) => (
  <>
    <Spacer shrinks={false} />
    <Header
      title={<FormattedMessage defaultMessage="Welcome to MLflow" description="Home page hero title" />}
    />
  </>
);
