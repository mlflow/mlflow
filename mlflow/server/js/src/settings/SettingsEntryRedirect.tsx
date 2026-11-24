import { useEffect } from 'react';
import { useNavigate } from '../common/utils/RoutingUtils';
import Routes from '../experiment-tracking/routes';
import { SETTINGS_SECTION_GENERAL } from './settingsSectionConstants';

/** `/settings` without a section segment redirects to `/settings/general`. */
const SettingsEntryRedirect = () => {
  const navigate = useNavigate();

  useEffect(() => {
    navigate(Routes.getSettingsSectionRoute(SETTINGS_SECTION_GENERAL), { replace: true });
  }, [navigate]);

  return null;
};

export default SettingsEntryRedirect;
