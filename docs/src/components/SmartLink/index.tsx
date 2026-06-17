import React from 'react';
import type { Props } from '@docusaurus/Link';
import OriginalLink from '@docusaurus-original/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

// Docusaurus is configured with `trailingSlash: true`, which appends a slash to
// `<Link>` targets. For a target that ends in a real file (e.g. the Sphinx-built
// API reference `rest-api.html`), this inserts a slash before the hash
// (`rest-api.html#foo` -> `rest-api.html/#foo`). The resulting path does not exist
// as an object on the S3-hosted site and returns an "AccessDenied" error.
//
// Such links point at static files, not Docusaurus routes, so render them as plain
// anchors. A plain <a> bypasses the trailing-slash normalization (matching the
// existing `APILink` component). Non-file links fall through to the real Link and
// keep client-side navigation. See https://github.com/mlflow/mlflow/issues/23966.
function isStaticFileLink(to: string | undefined): to is string {
  if (!to) {
    return false;
  }
  const [pathname] = to.split(/[#?]/);
  // The last path segment has a file extension (e.g. `rest-api.html`).
  return /\.[^/]+$/.test(pathname);
}

export default function Link(props: Props) {
  const { to, href, isNavLink, activeClassName, activeStyle, autoAddBaseUrl, ...anchorProps } = props as Props & {
    autoAddBaseUrl?: boolean;
  };
  const target = to ?? href;
  const isInternal = typeof target === 'string' && target.startsWith('/');
  // Hooks must run unconditionally; the result is only used for internal targets.
  const internalHref = useBaseUrl(isInternal ? (target as string) : '/');

  if (isStaticFileLink(target)) {
    return <a href={isInternal ? internalHref : target} {...anchorProps} />;
  }

  return <OriginalLink {...props} />;
}
