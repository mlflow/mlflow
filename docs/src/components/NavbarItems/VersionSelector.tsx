import React, { useState, useEffect } from 'react';
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { LinkLikeNavbarItemProps } from '@theme/NavbarItem';

interface VersionSelectorProps {
  mobile?: boolean;
  position?: 'left' | 'right';
  label?: string;
  [key: string]: any;
}

function getLabel(currentVersion: string, versions: string[]): string {
  if (currentVersion === 'latest' && versions.length > 0) {
    // version list is sorted in descending order, so the first one is the latest
    return `Version: ${versions[0]} (latest)`;
  }

  return `Version: ${currentVersion}`;
}

function VersionSelectorImpl({ mobile, label = 'Version', ...props }: VersionSelectorProps): JSX.Element {
  const [versions, setVersions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const versionsUrl = window.location.origin + '/docs/versions.json';

  // Determine current version from URL or default to latest
  const docPath = window.location.pathname;
  const currentVersion = docPath.match(/^\/docs\/([a-zA-Z0-9.]+)/)?.[1];

  const versionItems: LinkLikeNavbarItemProps[] = versions?.map((version) => ({
    type: 'default',
    label: version,
    to: window.location.origin + `/docs/${version}/`,
    target: '_self',
  }));

  useEffect(() => {
    const fetchVersions = async () => {
      try {
        const response = await fetch(versionsUrl);
        const data = await response.json();
        if (data['versions'] != null) {
          setVersions(data['versions']);
        }
      } catch (error) {
        // do nothing, this can happen in dev where the versions.json file is not available
      } finally {
        setLoading(false);
      }
    };

    fetchVersions();
  }, [versionsUrl]);

  if (loading || versions == null || versions.length === 0) {
    return null;
  }

  return (
    <DropdownNavbarItem {...props} mobile={mobile} label={getLabel(currentVersion, versions)} items={versionItems} />
  );
}

export default function VersionSelector(props: VersionSelectorProps): JSX.Element {
  return <BrowserOnly>{() => <VersionSelectorImpl {...props} />}</BrowserOnly>;
}
