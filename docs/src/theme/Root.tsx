import {
  DesignSystemProvider,
  DesignSystemThemeProvider,
} from "@databricks/design-system";
import { useCallback, useEffect, useState } from "react";

export default function Root({ children }: { children: React.ReactNode }) {
  const isDarkMode = useIsDarkMode();

  return (
    <>
      {/* @ts-ignore */}
      <DesignSystemThemeProvider isDarkMode={isDarkMode}>
        {/* @ts-ignore */}
        <DesignSystemProvider>{children}</DesignSystemProvider>
      </DesignSystemThemeProvider>
    </>
  );
}

function useIsDarkMode() {
  const getIsDarkMode = useCallback(
    () => document.querySelector("html").dataset.theme === "dark",
    []
  );
  const [isDarkMode, setIsDarkMode] = useState<boolean>(getIsDarkMode());

  const observer = new MutationObserver(() => {
    setIsDarkMode(getIsDarkMode());
  });

  useEffect(() => {
    observer.observe(document.querySelector("html"), {
      attributeFilter: ["data-theme"],
    });

    return () => observer.disconnect();
  });

  return isDarkMode;
}
