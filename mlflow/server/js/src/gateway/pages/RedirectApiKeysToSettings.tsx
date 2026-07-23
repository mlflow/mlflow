import { useEffect } from 'react';
import { useNavigate } from '../../common/utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';
import {
  SETTINGS_RETURN_TO_PARAM,
  SETTINGS_SECTION_GENERAL,
  SETTINGS_SECTION_LLM_CONNECTIONS,
} from '../../settings/settingsSectionConstants';

/**
 * Legacy `/gateway/api-keys` route: API keys now live as the LLM Connections
 * section within Settings > General.
 */
const RedirectApiKeysToSettings = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const settingsRoute = Routes.getSettingsSectionRoute(SETTINGS_SECTION_GENERAL);
    navigate(
      `${settingsRoute}?${SETTINGS_RETURN_TO_PARAM}=${encodeURIComponent('/gateway')}#${SETTINGS_SECTION_LLM_CONNECTIONS}`,
      { replace: true },
    );
  }, [navigate]);

  return null;
};

export default RedirectApiKeysToSettings;
