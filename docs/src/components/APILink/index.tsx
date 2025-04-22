import React from "react";
import APIModules from "../../api_modules.json";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Link from "@docusaurus/Link";

const getModule = (fn: string) => {
  const parts = fn.split(".");
  // find the longest matching module
  for (let i = parts.length; i > 0; i--) {
    const module = parts.slice(0, i).join(".");
    if (APIModules[module]) {
      return module;
    }
  }

  return null;
};

export function APILink({
  fn,
  baseUrl,
  children,
}: {
  fn: string;
  baseUrl?: string;
  children?: React.ReactNode;
}) {
  const module = getModule(fn);

  if (!module) {
    return <>{children}</>;
  }

  const relativePath = `/${APIModules[module]}#${fn}`;
  const docLink = baseUrl ? `${baseUrl}${relativePath}` : useBaseUrl(relativePath);
  return (
    <Link to={docLink} target="_blank">
      {children ?? <code>{fn}()</code>}
    </Link>
  );
}
