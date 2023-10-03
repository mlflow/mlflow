import React from 'react';
import {
  Button,
  CloseIcon,
  CopyIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { CopyButton } from '../../shared/building_blocks/CopyButton';

const PREVIEW_SIDEBAR_WIDTH = 300;

/**
 * Displays a sidebar helpful in expanding textual data in table components.
 * Will be replaced by DuBois standardized component in the future.
 */
export const PreviewSidebar = ({
  content,
  copyText,
  headerText,
  empty,
  onClose,
}: {
  content?: React.ReactNode;
  copyText?: string;
  headerText?: string;
  empty?: React.ReactNode;
  onClose?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        width: PREVIEW_SIDEBAR_WIDTH,
        padding: theme.spacing.sm,
        paddingRight: 0,
        borderLeft: `1px solid ${theme.colors.borderDecorative}`,
        overflow: 'auto',
        height: '100%',
      }}
      data-testid='preview-sidebar-content'
    >
      {content ? (
        <>
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: '1fr auto auto',
              rowGap: theme.spacing.sm,
              alignItems: 'center',
            }}
          >
            {headerText && <Typography.Title level={4}>{headerText}</Typography.Title>}
            {copyText && <CopyButton copyText={copyText} showLabel={false} icon={<CopyIcon />} />}
            {onClose && <Button type='primary' icon={<CloseIcon />} onClick={onClose} />}
          </div>
          <div
            css={{
              // Preserve original line breaks
              whiteSpace: 'pre-wrap',
            }}
          >
            {content}
          </div>
        </>
      ) : (
        <div css={{ marginTop: theme.spacing.md }}>{empty}</div>
      )}
    </div>
  );
};
