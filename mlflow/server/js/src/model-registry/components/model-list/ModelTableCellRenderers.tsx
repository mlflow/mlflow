import { FormattedMessage } from 'react-intl';
import { Link } from 'react-router-dom';
import { getModelVersionPageRoute } from '../../routes';

const EmptyCell = () => <>&mdash;</>;

/**
 * Renders model version with the link in the models table
 */
export const ModelListVersionLinkCell = ({
  versionNumber,
  name,
}: {
  versionNumber?: string;
  name: string;
}) => {
  if (!versionNumber) {
    return <EmptyCell />;
  }
  return (
    <FormattedMessage
      defaultMessage='<link>Version {versionNumber}</link>'
      description='Row entry for version columns in the registered model page'
      values={{
        versionNumber,
        link: (text) => <Link to={getModelVersionPageRoute(name, versionNumber)}>{text}</Link>,
      }}
    />
  );
};
