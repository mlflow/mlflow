import {
  DesignSystemProvider,
  DesignSystemThemeProvider,
} from "@databricks/design-system";
import useIsBrowser from "@docusaurus/useIsBrowser";
import { useEffect, useState } from "react";

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

function getIsDarkMode() {
  return document.querySelector("html").dataset.theme === "dark";
}

function useIsDarkMode() {
  const isBrowser = useIsBrowser();
  const initialIsDarkMode = isBrowser ? getIsDarkMode() : false;
  const [isDarkMode, setIsDarkMode] = useState<boolean>(initialIsDarkMode);

  useEffect(() => {
    if (!isBrowser) return;

    const observer = new MutationObserver(() => {
      setIsDarkMode(getIsDarkMode());
    });

    observer.observe(document.querySelector("html"), {
      attributeFilter: ["data-theme"],
    });

    return () => observer.disconnect();
  }, [isBrowser]);

  return isDarkMode;
}
