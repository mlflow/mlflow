/**
 * Dev-only floating toolbar for switching the currently authenticated user.
 *
 * Enable: localStorage.setItem('mlflow.settings.admin.enable-dev-user-switcher_v1', 'true')
 *
 * How it works:
 *   MLflow auth is HTTP Basic, and FetchUtils.getDefaultHeadersFromCookies picks up any
 *   cookie prefixed `mlflow-request-header-` and forwards it as a request header on every
 *   fetch. So we set `mlflow-request-header-Authorization=Basic <base64(user:password)>`
 *   and every subsequent request authenticates as the new user — no interceptor needed.
 *
 * Recovery:
 *   The cookie-forwarding mechanism means a wrong password locks every subsequent request
 *   into 401. To avoid the trap we validate creds against `/users/current` before writing
 *   the cookie. The "Clear" button in the panel header is the manual escape hatch — it
 *   wipes the cookies + stored creds and reloads.
 *
 * Listing users requires admin. For non-admin sessions the panel falls back to two rows
 * (`admin` and the current user) so you can always switch back. Passwords are stored in
 * localStorage (`admin.dev-user-credentials`) as a dev convenience.
 */
import { useCallback, useState } from 'react';
import { Alert, Button, Input, Modal, Spinner, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from '../common/utils/FetchUtils';
import { getLocalStorageItem, setLocalStorageItem } from '../shared/web-shared/hooks/useLocalStorage';
import { useUsersQuery } from './hooks';

const CREDENTIALS_STORAGE_KEY = 'admin.dev-user-credentials';
const CREDENTIALS_STORAGE_VERSION = 1;
const MLFLOW_USER_COOKIE = 'mlflow_user';
const AUTH_HEADER_COOKIE = 'mlflow-request-header-Authorization';
const FALLBACK_ADMIN_USERNAME = 'admin';

const getStoredCredentials = (): Record<string, string> =>
  getLocalStorageItem<Record<string, string>>(CREDENTIALS_STORAGE_KEY, CREDENTIALS_STORAGE_VERSION, false, {});

const setStoredCredential = (username: string, password: string) => {
  const map = getStoredCredentials();
  map[username] = password;
  setLocalStorageItem(CREDENTIALS_STORAGE_KEY, CREDENTIALS_STORAGE_VERSION, false, map);
};

const getCurrentUsername = (): string =>
  document.cookie
    .split('; ')
    .find((row) => row.startsWith(`${MLFLOW_USER_COOKIE}=`))
    ?.substring(`${MLFLOW_USER_COOKIE}=`.length) ?? '';

const applyCredentials = (username: string, password: string) => {
  const encoded = btoa(`${username}:${password}`);
  // Cookie values containing spaces are not RFC 6265-compliant and can be
  // truncated by some parsers. URI-encode the value so it round-trips through
  // ``cookie.parse`` in ``getDefaultHeadersFromCookies`` (which decodes by
  // default), reconstituting ``Basic <base64>`` for the Authorization header.
  document.cookie = `${AUTH_HEADER_COOKIE}=${encodeURIComponent(`Basic ${encoded}`)}; path=/`;
  document.cookie = `${MLFLOW_USER_COOKIE}=${username}; path=/`;
};

const clearAuthCookies = () => {
  const expired = 'expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
  document.cookie = `${AUTH_HEADER_COOKIE}=; ${expired}`;
  document.cookie = `${MLFLOW_USER_COOKIE}=; ${expired}`;
};

// Validate creds against /users/current with an explicit Authorization header,
// using plain `fetch` (not `fetchEndpoint`) so the existing cookie-forwarded
// header doesn't override ours. Returns true on 200, false otherwise.
const validateCredentials = async (username: string, password: string): Promise<boolean> => {
  try {
    const url = getAjaxUrl('ajax-api/2.0/mlflow/users/current');
    const response = await fetch(url, {
      method: 'GET',
      headers: { Authorization: `Basic ${btoa(`${username}:${password}`)}` },
    });
    return response.ok;
  } catch {
    return false;
  }
};

export const DevUserSwitcher = () => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useUsersQuery();
  const [passwordPromptFor, setPasswordPromptFor] = useState<string | null>(null);
  const [passwordInput, setPasswordInput] = useState('');
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  // Cookies are not reactive — track the active user in state so the highlight
  // updates immediately after a switch (before queries finish refetching).
  const [currentUsername, setCurrentUsername] = useState<string>(getCurrentUsername);

  const users = data?.users ?? [];

  const completeSwitch = useCallback(
    (username: string, password: string) => {
      applyCredentials(username, password);
      setCurrentUsername(username);
      queryClient.invalidateQueries();
    },
    [queryClient],
  );

  const handleSwitch = useCallback(
    (username: string, options: { forcePrompt?: boolean } = {}) => {
      const stored = getStoredCredentials();
      const password = stored[username];
      if (password && !options.forcePrompt) {
        completeSwitch(username, password);
      } else {
        setPasswordInput('');
        setPasswordError(null);
        setPasswordPromptFor(username);
      }
    },
    [completeSwitch],
  );

  const handlePasswordSubmit = async () => {
    if (!passwordPromptFor || !passwordInput || validating) return;
    setValidating(true);
    setPasswordError(null);
    const ok = await validateCredentials(passwordPromptFor, passwordInput);
    setValidating(false);
    if (!ok) {
      setPasswordError('Authentication failed. Check the password and try again.');
      return;
    }
    setStoredCredential(passwordPromptFor, passwordInput);
    completeSwitch(passwordPromptFor, passwordInput);
    setPasswordPromptFor(null);
    setPasswordInput('');
  };

  const handleClearCreds = () => {
    clearAuthCookies();
    setLocalStorageItem(CREDENTIALS_STORAGE_KEY, CREDENTIALS_STORAGE_VERSION, false, {});
    queryClient.clear();
    window.location.reload();
  };

  return (
    <>
      <div
        css={{
          position: 'fixed',
          bottom: theme.spacing.md,
          right: theme.spacing.md,
          zIndex: 9999,
          background: theme.colors.backgroundPrimary,
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.sm,
          boxShadow: theme.general.shadowLow,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          minWidth: 200,
          maxWidth: 280,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text bold size="sm" css={{ flex: 1 }}>
            Dev User Switcher
          </Typography.Text>
          <Button
            componentId="dev.user_switcher.clear"
            size="small"
            type="link"
            onClick={handleClearCreds}
            title="Clear stored switcher creds and the auth cookie, then reload. Use to recover from a stuck Basic Auth state."
          >
            Clear
          </Button>
        </div>
        {isLoading && (
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <Spinner size="small" />
            <Typography.Text size="sm" color="secondary">
              Loading users…
            </Typography.Text>
          </div>
        )}
        {error && (
          <Alert
            componentId="dev.user_switcher.error"
            type="warning"
            message="Can't list users"
            description="Requires admin access. Switch to admin below to recover."
          />
        )}
        {!isLoading && !error && users.length === 0 && (
          <Typography.Text size="sm" color="secondary">
            No users found.
          </Typography.Text>
        )}
        {(error
          ? [
              { username: FALLBACK_ADMIN_USERNAME, is_admin: true },
              ...(currentUsername && currentUsername !== FALLBACK_ADMIN_USERNAME
                ? [{ username: currentUsername, is_admin: false }]
                : []),
            ]
          : users
        ).map((user) => {
          const isActive = user.username === currentUsername;
          const isFallback = Boolean(error);
          return (
            <div
              key={user.username}
              role="button"
              tabIndex={0}
              onClick={() => handleSwitch(user.username, { forcePrompt: isFallback })}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') handleSwitch(user.username, { forcePrompt: isFallback });
              }}
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderRadius: theme.borders.borderRadiusMd - 2,
                cursor: 'pointer',
                background: isActive ? theme.colors.actionPrimaryBackgroundDefault : 'transparent',
                color: isActive ? theme.colors.actionPrimaryTextDefault : theme.colors.textPrimary,
                '&:hover': {
                  background: isActive
                    ? theme.colors.actionPrimaryBackgroundHover
                    : theme.colors.actionDefaultBackgroundHover,
                },
              }}
            >
              <Typography.Text size="sm" css={{ color: 'inherit' }}>
                {user.username}
              </Typography.Text>
              {user.is_admin && (
                <Tag componentId="dev.user_switcher.admin_tag" color="indigo" css={{ marginLeft: 'auto' }}>
                  admin
                </Tag>
              )}
            </div>
          );
        })}
      </div>
      <Modal
        componentId="dev.user_switcher.password_modal"
        title={`Password for ${passwordPromptFor}`}
        visible={Boolean(passwordPromptFor)}
        onCancel={() => {
          setPasswordPromptFor(null);
          setPasswordInput('');
          setPasswordError(null);
        }}
        onOk={handlePasswordSubmit}
        okText="Switch"
        confirmLoading={validating}
        okButtonProps={{ disabled: !passwordInput || validating }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text>
            Enter the password for <strong>{passwordPromptFor}</strong>. It will be remembered in localStorage for
            future switches on this machine.
          </Typography.Text>
          <Input
            componentId="dev.user_switcher.password_input"
            type="password"
            value={passwordInput}
            onChange={(e) => {
              setPasswordInput(e.target.value);
              if (passwordError) setPasswordError(null);
            }}
            onPressEnter={handlePasswordSubmit}
            placeholder="Password"
            autoFocus
          />
          {passwordError && (
            <Alert
              componentId="dev.user_switcher.password_error"
              type="error"
              message={passwordError}
              closable={false}
            />
          )}
        </div>
      </Modal>
    </>
  );
};
