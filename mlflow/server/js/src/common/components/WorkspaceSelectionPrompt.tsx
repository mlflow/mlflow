import React from 'react';
import { Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { WorkspaceSelector } from './WorkspaceSelector';

interface WorkspaceSelectionPromptProps {
  isOpen: boolean;
  onClose: () => void;
}

export const WorkspaceSelectionPrompt: React.FC<WorkspaceSelectionPromptProps> = ({ isOpen, onClose }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Modal
      componentId="workspace_selection_prompt"
      visible={isOpen}
      onCancel={onClose}
      footer={null}
      title={
        <FormattedMessage
          defaultMessage="Select a Workspace"
          description="Workspace selection prompt modal title"
        />
      }
    >
      <div css={{ 
        display: 'flex', 
        flexDirection: 'column', 
        gap: theme.spacing.md,
        padding: theme.spacing.sm,
      }}>
        <p css={{ 
          margin: 0, 
          color: theme.colors.textSecondary,
          fontSize: theme.typography.fontSizeMd,
        }}>
          <FormattedMessage
            defaultMessage="This MLflow instance uses workspaces to organize experiments. Please select a workspace to continue."
            description="Workspace selection prompt modal description"
          />
        </p>
        <WorkspaceSelector />
      </div>
    </Modal>
  );
};

