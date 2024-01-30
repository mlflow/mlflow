import { Accordion, useDesignSystemTheme } from '@databricks/design-system';
import { CSSObject, css } from '@emotion/react';
import { useMemo } from 'react';

interface MetricChartsAccordionProps {
  activeKey?: string | string[];
  onActiveKeyChange?: (key: string | string[]) => void;
  children: React.ReactNode;
}

const MetricChartsAccordion = ({ activeKey, onActiveKeyChange, children }: MetricChartsAccordionProps) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('collapse');

  const styles = useMemo(() => {
    const classItem = `.${clsPrefix}-item`;
    const classItemActive = `${classItem}-active`;
    const classHeader = `.${clsPrefix}-header`;
    const classContent = `.${clsPrefix}-content`;
    const classContentBox = `.${clsPrefix}-content-box`;
    const classArrow = `.${clsPrefix}-arrow`;

    const styles: CSSObject = {
      [classContent]: {
        padding: '0px !important',
        backgroundColor: 'transparent !important',
      },

      [classContentBox]: {
        padding: '0 0 12px 0px !important',
        backgroundColor: 'transparent !important',
      },

      [`& > ${classItem} > ${classHeader} > ${classArrow}`]: {
        fontSize: theme.general.iconSize,
        left: 12,
        // TODO: This is needed currently because the rotated icon isn't centered, remove when accordion is fixed
        verticalAlign: '-7px',
        transform: 'rotate(-90deg)',
      },

      [`& > ${classItemActive} > ${classHeader} > ${classArrow}`]: {
        transform: 'rotate(0deg)',
      },

      [classHeader]: {
        display: 'flex',
        color: theme.colors.textPrimary,
        fontWeight: 600,
        alignItems: 'center',

        '&:focus-visible': {
          outlineColor: `${theme.colors.primary} !important`,
          outlineStyle: 'auto !important',
        },
      },

      [`& > ${classItem}`]: {
        borderBottom: `1px solid ${theme.colors.border}`,
        borderRadius: 0,
      },

      [`& > ${classItem} > ${classHeader}`]: {
        padding: 0,
        lineHeight: '20px',
        height: theme.general.heightBase,
      },
    };
    return styles;
  }, [theme, clsPrefix]);

  if (activeKey && onActiveKeyChange) {
    return (
      <Accordion
        activeKey={activeKey}
        onChange={onActiveKeyChange}
        dangerouslyAppendEmotionCSS={css(styles)}
        dangerouslySetAntdProps={{ expandIconPosition: 'left' }}
      >
        {children}
      </Accordion>
    );
  } else {
    return (
      <Accordion dangerouslyAppendEmotionCSS={css(styles)} dangerouslySetAntdProps={{ expandIconPosition: 'left' }}>
        {children}
      </Accordion>
    );
  }
};

export default MetricChartsAccordion;
