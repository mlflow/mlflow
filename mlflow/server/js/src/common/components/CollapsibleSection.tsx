import React, { useCallback } from 'react';
import { SectionErrorBoundary } from './error-boundaries/SectionErrorBoundary';
import type { DesignSystemThemeInterface } from '@databricks/design-system';
import { ChevronRightIcon, useDesignSystemTheme, Accordion, importantify } from '@databricks/design-system';
import { useIntl } from 'react-intl';

interface CollapsibleSectionProps {
  title: string | any;
  forceOpen?: boolean;
  children: React.ReactNode;
  showServerError?: boolean;
  defaultCollapsed?: boolean;
  onChange?: (key: string | string[]) => void;
  className?: string;
  componentId?: string;
}

// Custom styles to make <Accordion> look like previously used <Collapse> from antd
const getAccordionStyles = ({
  theme,
  getPrefixedClassName,
}: Pick<DesignSystemThemeInterface, 'theme' | 'getPrefixedClassName'>) => {
  const clsPrefix = getPrefixedClassName('collapse');

  const classItem = `.${clsPrefix}-item`;
  const classHeader = `.${clsPrefix}-header`;
  const classContentBox = `.${clsPrefix}-content-box`;

  return {
    fontSize: 14,
    [`& > ${classItem} > ${classHeader}`]: {
      paddingLeft: 0,
      paddingTop: 12,
      paddingBottom: 12,
      display: 'flex',
      alignItems: 'center',
      fontSize: 16,
      fontWeight: 'normal',
      lineHeight: theme.typography.lineHeightLg,
    },
    [classContentBox]: {
      padding: `${theme.spacing.xs}px 0 ${theme.spacing.md}px 0`,
    },
  };
};

export function CollapsibleSection(props: CollapsibleSectionProps) {
  const {
    title,
    forceOpen,
    showServerError,
    defaultCollapsed,
    onChange,
    className,
    componentId = 'mlflow.common.generic_collapsible_section',
  } = props;

  // We need to spread `activeKey` into <Collapse/> as an optional prop because its enumerability
  // affects rendering, i.e. passing `activeKey={undefined}` is different from not passing activeKey
  const activeKeyProp = forceOpen && { activeKey: ['1'] };
  const defaultActiveKey = defaultCollapsed ? null : ['1'];

  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  const getExpandIcon = useCallback(
    ({ isActive }: { isActive?: boolean }) => (
      <div
        css={importantify({ width: theme.general.heightBase / 2, transform: isActive ? 'rotate(90deg)' : undefined })}
      >
        <ChevronRightIcon
          css={{
            svg: { width: theme.general.heightBase / 2, height: theme.general.heightBase / 2 },
          }}
          aria-label={
            isActive
              ? formatMessage(
                  {
                    defaultMessage: 'collapse {title}',
                    description: 'Common component > collapsible section > alternative label when expand',
                  },
                  { title },
                )
              : formatMessage(
                  {
                    defaultMessage: 'expand {title}',
                    description: 'Common component > collapsible section > alternative label when collapsed',
                  },
                  { title },
                )
          }
        />
      </div>
    ),
    [theme, title, formatMessage],
  );

  return (
    <Accordion
      componentId={componentId}
      {...activeKeyProp}
      dangerouslyAppendEmotionCSS={getAccordionStyles({ theme, getPrefixedClassName })}
      dangerouslySetAntdProps={{
        className,
        expandIconPosition: 'left',
        expandIcon: getExpandIcon,
      }}
      defaultActiveKey={defaultActiveKey ?? undefined}
      onChange={onChange}
    >
      <Accordion.Panel header={title} key="1">
        <SectionErrorBoundary showServerError={showServerError}>{props.children}</SectionErrorBoundary>
      </Accordion.Panel>
    </Accordion>
  );
}
