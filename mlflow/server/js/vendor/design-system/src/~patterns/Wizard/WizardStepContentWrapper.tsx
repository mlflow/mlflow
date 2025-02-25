import type { PropsWithChildren } from 'react';
import React from 'react';

import { getShadowScrollStyles, Typography, useDesignSystemTheme } from '../../design-system';

export interface WizardStepContentWrapperProps {
  /**
   * Displayed in header of the step.
   *
   * ex)
   * `<FormattedMessage
   *   defaultMessage="STEP{stepIndex}"
   *   description="Wizard step number"
   *   values={{ stepIndex: stepIndex + 1 }}
   * />`
   */
  header: React.ReactNode;

  /**
   * Title of the step
   */
  title: React.ReactNode;

  /**
   * Description of the step displayed below the step title
   */
  description: React.ReactNode;
}

export function WizardStepContentWrapper({
  header,
  title,
  description,
  children,
}: PropsWithChildren<WizardStepContentWrapperProps>) {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      <div
        style={{
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.lg,
          display: 'flex',
          flexDirection: 'column',
          borderTopLeftRadius: theme.legacyBorders.borderRadiusLg,
          borderTopRightRadius: theme.legacyBorders.borderRadiusLg,
        }}
      >
        <Typography.Text size="sm" style={{ fontWeight: 500 }}>
          {header}
        </Typography.Text>
        <Typography.Title withoutMargins style={{ paddingTop: theme.spacing.lg }} level={3}>
          {title}
        </Typography.Title>
        <Typography.Text color="secondary">{description}</Typography.Text>
      </div>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          padding: `${theme.spacing.lg}px ${theme.spacing.lg}px 0`,
          overflowY: 'auto',
          ...getShadowScrollStyles(theme),
        }}
      >
        {children}
      </div>
    </div>
  );
}
