import React, { createContext, useCallback, useContext } from 'react';
import type { AuthUser, AuthProvider as AuthProviderInfo } from '../../experiment-tracking/hooks/useServerInfo';
import { useAuthInfo } from '../../experiment-tracking/hooks/useServerInfo';

interface AuthContextValue {
  isAuthenticated: boolean;
  authType: 'oauth' | 'basic' | 'none';
  user: AuthUser | null;
  providers: AuthProviderInfo[];
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  isAuthenticated: false,
  authType: 'none',
  user: null,
  providers: [],
  logout: async () => {},
});

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const { authType, authUser, authProviders } = useAuthInfo();

  const logout = useCallback(async () => {
    try {
      const response = await fetch('/auth/logout', {
        method: 'POST',
        credentials: 'same-origin',
      });
      const data = await response.json();
      window.location.href = data.redirect_url || '/auth/login';
    } catch {
      window.location.href = '/auth/login';
    }
  }, []);

  const value: AuthContextValue = {
    isAuthenticated: authType !== 'none' && authUser !== null && authUser !== undefined,
    authType: authType as 'oauth' | 'basic' | 'none',
    user: authUser,
    providers: authProviders,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuthContext = () => useContext(AuthContext);
