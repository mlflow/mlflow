import { Drawer, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { CSSObject } from '@emotion/react';
import type { ReactNode } from 'react';

export type HomeQuickstartFrameworkOption<T extends string> = {
  id: T;
  label: string;
  logo?: string;
  selectedLogo?: string;
  buttonComponentId?: string;
};

export type HomeQuickstartDrawerProps<T extends string> = {
  componentId: string;
  icon: ReactNode;
  title: ReactNode;
  intro: ReactNode;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  frameworks: HomeQuickstartFrameworkOption<T>[];
  selectedFramework: T;
  onSelectFramework: (framework: T) => void;
  children: ReactNode;
  leftFooter?: ReactNode;
  contentCss?: CSSObject;
};

export const HomeQuickstartDrawer = <T extends string>({
  componentId,
  icon,
  title,
  intro,
  isOpen,
  onOpenChange,
  frameworks,
  selectedFramework,
  onSelectFramework,
  children,
  leftFooter,
  contentCss,
}: HomeQuickstartDrawerProps<T>) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Drawer.Root modal open={isOpen} onOpenChange={onOpenChange}>
      <Drawer.Content
        componentId={componentId}
        width="70vw"
        title={
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
            }}
          >
            <span
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                background: theme.colors.actionDefaultBackgroundHover,
                padding: theme.spacing.xs,
                color: theme.colors.blue500,
                height: 'min-content',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {icon}
            </span>
            {title}
          </span>
        }
      >
        <Typography.Text color="secondary">{intro}</Typography.Text>
        <Spacer size="md" />
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            gap: theme.spacing.lg,
            minHeight: 0,
          }}
        >
          <aside
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
              minWidth: 220,
              maxWidth: 260,
            }}
          >
            {frameworks.map(({ id, label, logo, selectedLogo, buttonComponentId }) => {
              const isSelected = id === selectedFramework;
              const logoSrc = isSelected && selectedLogo ? selectedLogo : logo;

              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => onSelectFramework(id)}
                  data-component-id={buttonComponentId}
                  aria-pressed={isSelected}
                  css={{
                    border: 0,
                    borderRadius: theme.borders.borderRadiusSm,
                    padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                    textAlign: 'left' as const,
                    cursor: 'pointer',
                    backgroundColor: isSelected
                      ? theme.colors.actionPrimaryBackgroundDefault
                      : theme.colors.backgroundSecondary,
                    color: isSelected ? theme.colors.actionPrimaryTextDefault : theme.colors.textPrimary,
                    boxShadow: theme.shadows.sm,
                    transition: 'background 150ms ease',
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.sm,
                    '&:hover': {
                      backgroundColor: isSelected
                        ? theme.colors.actionPrimaryBackgroundHover
                        : theme.colors.actionDefaultBackgroundHover,
                    },
                    '&:focus-visible': {
                      outline: `2px solid ${theme.colors.actionPrimaryBackgroundHover}`,
                      outlineOffset: 2,
                    },
                  }}
                >
                  {logoSrc && (
                    <img src={logoSrc} width={28} height={28} alt="" aria-hidden css={{ display: 'block' }} />
                  )}
                  {label}
                </button>
              );
            })}
            {leftFooter && (
              <>
                <Spacer size="sm" />
                {leftFooter}
              </>
            )}
          </aside>
          <div
            css={{
              flex: 1,
              minWidth: 0,
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.lg,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusLg,
              padding: theme.spacing.lg,
              backgroundColor: theme.colors.backgroundPrimary,
              boxShadow: theme.shadows.xs,
              ...contentCss,
            }}
          >
            {children}
          </div>
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};

export default HomeQuickstartDrawer;
