import React, { createContext, type ReactNode, useContext } from 'react';

const TracesViewTableNoTracesQuickstartContext = createContext<{
  introductionText?: ReactNode;
  displayVersionWarnings?: boolean;
}>({});

/**
 * Allows to alter default behavior of a quickstart tutorial for logging traces
 */
export const TracesViewTableNoTracesQuickstartContextProvider = ({
  children,
  introductionText,
  displayVersionWarnings,
}: {
  children: ReactNode;
  introductionText?: ReactNode;
  displayVersionWarnings?: boolean;
}) => {
  return (
    <TracesViewTableNoTracesQuickstartContext.Provider value={{ introductionText, displayVersionWarnings }}>
      {children}
    </TracesViewTableNoTracesQuickstartContext.Provider>
  );
};

export const useTracesViewTableNoTracesQuickstartContext = () => useContext(TracesViewTableNoTracesQuickstartContext);
