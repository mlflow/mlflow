import { useEffect } from 'react';
import { useNavigate } from '../../common/utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';
import { SETTINGS_RETURN_TO_PARAM, SETTINGS_SECTION_LLM_CONNECTIONS } from '../../settings/settingsSectionConstants';

/**
 * Legacy `/gateway/api-keys` route: API keys live under Settings > LLM Connections.
 */
const RedirectApiKeysToSettings = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const settingsRoute = Routes.getSettingsSectionRoute(SETTINGS_SECTION_LLM_CONNECTIONS);
    navigate(`${settingsRoute}?${SETTINGS_RETURN_TO_PARAM}=${encodeURIComponent('/gateway')}`, { replace: true });
  }, [navigate]);

  return null;
};

export default RedirectApiKeysToSettings;
