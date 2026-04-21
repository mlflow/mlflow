import { useEffect } from 'react';
import { useNavigate } from '../common/utils/RoutingUtils';
import Routes from '../experiment-tracking/routes';

/** `/settings` without a section segment redirects to `/settings/general`. */
const SettingsEntryRedirect = () => {
  const navigate = useNavigate();

  useEffect(() => {
    navigate(Routes.getSettingsSectionRoute('general'), { replace: true });
  }, [navigate]);

  return null;
};

export default SettingsEntryRedirect;
