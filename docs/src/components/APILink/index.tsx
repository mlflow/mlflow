import React from 'react';
import APIModules from '../../api_modules.json';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Link from '@docusaurus/Link';

const getModule = (fn: string) => {
  const parts = fn.split('.');
  // find the longest matching module
  for (let i = parts.length; i > 0; i--) {
    const module = parts.slice(0, i).join('.');
    if (APIModules[module]) {
      return module;
    }
  }

  return null;
};

export function APILink({ fn, children, hash }: { fn: string; children?: React.ReactNode; hash?: string }) {
  const module = getModule(fn);

  if (!module) {
    return <>{children}</>;
  }

  const docLink = useBaseUrl(`/${APIModules[module]}#${hash ?? fn}`);
  return (
    <a href={docLink} target="_blank">
      {children ?? <code>{fn}()</code>}
    </a>
  );
}
