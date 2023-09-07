import React from 'react';
import { CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
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
}: {
  content?: React.ReactNode;
  copyText?: string;
  headerText?: string;
  empty?: React.ReactNode;
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
    >
      {content ? (
        <>
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: '1fr auto',
              rowGap: theme.spacing.sm,
              alignItems: 'center',
            }}
          >
            {headerText && <Typography.Title level={4}>{headerText}</Typography.Title>}
            {copyText && <CopyButton copyText={copyText} showLabel={false} icon={<CopyIcon />} />}
          </div>

          {content}
        </>
      ) : (
        <div css={{ marginTop: theme.spacing.md }}>{empty}</div>
      )}
    </div>
  );
};
