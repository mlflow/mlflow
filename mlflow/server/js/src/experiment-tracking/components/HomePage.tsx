import { Navigate } from '../../common/utils/RoutingUtils';
import Routes from '../routes';

const HomePage = () => {
  return <Navigate to={Routes.experimentsObservatoryRoute} />;
};

export default HomePage;
