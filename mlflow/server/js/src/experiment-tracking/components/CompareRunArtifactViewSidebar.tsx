import React, { useState } from 'react';
import { getExtension, IMAGE_EXTENSIONS, DATA_EXTENSIONS, TEXT_EXTENSIONS } from '../../common/utils/FileUtils';
import { useDesignSystemTheme } from '@databricks/design-system';
import { Theme } from '@emotion/react';

interface CompareRunArtifactViewSidebarProps {
  artifacts: string[];
  onSelectArtifact: (artifactPath: string) => void;
}

export const CompareRunArtifactViewSidebar = ({ artifacts, onSelectArtifact }: CompareRunArtifactViewSidebarProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedArtifact, setSelectedArtifact] = useState<string | null>(null);

  const handleSelect = (artifactPath: string) => {
    setSelectedArtifact(artifactPath);
    onSelectArtifact(artifactPath);
  };

  const getIconClass = (artifact: string) => {
    const extension = getExtension(artifact);
    if (IMAGE_EXTENSIONS.has(extension)) return 'fa fa-file-image-o';
    if (DATA_EXTENSIONS.has(extension)) return 'fa fa-file-excel-o';
    if (TEXT_EXTENSIONS.has(extension)) return 'fa fa-file-code-o';
    return 'fa fa-file-text-o';
  };

  return (
    <div style={getListViewStyle(theme)}>
      {artifacts.map((artifact) => (
        <div
          key={artifact}
          className={`artifact-list-item ${selectedArtifact === artifact ? 'selected' : ''}`}
          onClick={() => handleSelect(artifact)}
          role="button"
          style={{
            ...getItemStyle(theme),
            ...(selectedArtifact === artifact ? getSelectedStyle(theme) : {}),
          }}
        >
          <i className={getIconClass(artifact)} style={{ marginRight: '5px' }} />
          {artifact}
        </div>
      ))}
    </div>
  );
};

const getListViewStyle = (theme: Theme) => ({
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textPrimary,
  flex: '1 1 0%',
  whiteSpace: 'nowrap',
  border: `1px solid ${theme.colors.grey300}`,
  height: '100%',
  overflowY: 'auto',
});

const getItemStyle = (theme: Theme) => ({
  padding: '5px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  borderBottom: `1px solid ${theme.colors.grey300}`,
});

const getSelectedStyle = (theme: Theme) => ({
  backgroundColor: theme.isDarkMode ? theme.colors.grey700 : theme.colors.grey300,
});
