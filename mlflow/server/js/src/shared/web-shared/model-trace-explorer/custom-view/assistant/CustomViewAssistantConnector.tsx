import type { ReactNode } from 'react';
import { createContext, useContext } from 'react';

/** Options for opening the assistant chat panel. */
export type OpenCustomViewAssistantOptions = {
  // Submit the prompt in a FRESH assistant session (thread). Set only when
  // building a brand-new custom view; edits reuse the current session so the
  // conversation continues.
  newSession?: boolean;
};

/**
 * Connects the Custom View to the host application's agent (MLflow Assistant).
 * The host app provides a way to open the assistant chat panel and optionally
 * submit a prompt immediately, and reports whether the assistant is currently
 * streaming.
 */
export type CustomViewAssistantConnector = {
  openAssistant?: (prompt?: string, options?: OpenCustomViewAssistantOptions) => void;
  isStreaming?: boolean;
  isPending?: boolean;
};

const CustomViewAssistantConnectorContext = createContext<CustomViewAssistantConnector>({});

export const CustomViewAssistantConnectorProvider = ({
  connector,
  children,
}: {
  connector: CustomViewAssistantConnector;
  children: ReactNode;
}): JSX.Element => (
  <CustomViewAssistantConnectorContext.Provider value={connector}>
    {children}
  </CustomViewAssistantConnectorContext.Provider>
);

export const useCustomViewAssistantConnector = (): CustomViewAssistantConnector =>
  useContext(CustomViewAssistantConnectorContext);
