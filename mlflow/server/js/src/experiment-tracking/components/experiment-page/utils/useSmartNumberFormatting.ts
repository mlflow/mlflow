import { useLocalStorage } from '@databricks/web-shared/hooks';

export const SMART_NUMBER_FORMATTING_KEY = 'mlflow.settings.smartNumberFormatting';
export const SMART_NUMBER_FORMATTING_VERSION = 1;

/**
 * Returns whether column-aware smart number formatting is enabled.
 * Reads from localStorage so it stays in sync with the Settings page toggle.
 * Defaults to true (enabled).
 */
export const useSmartNumberFormattingEnabled = (): boolean => {
  const [isEnabled] = useLocalStorage({
    key: SMART_NUMBER_FORMATTING_KEY,
    version: SMART_NUMBER_FORMATTING_VERSION,
    initialValue: true,
  });
  return isEnabled ?? true;
};
