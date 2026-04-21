import { useEffect } from 'react';
import { useNavigate } from '../../common/utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';
import { SETTINGS_SECTION_LLM_CONNECTIONS } from '../../settings/settingsSectionConstants';

/**
 * Legacy `/gateway/api-keys` route: API keys live under Settings > LLM Connections.
 */
const RedirectApiKeysToSettings = () => {
  const navigate = useNavigate();

  useEffect(() => {
    navigate(Routes.getSettingsSectionRoute(SETTINGS_SECTION_LLM_CONNECTIONS), { replace: true });
  }, [navigate]);

  return null;
};

export default RedirectApiKeysToSettings;
