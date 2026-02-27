import { Accordion, useDesignSystemTheme } from '@databricks/design-system';
import type { CSSObject } from '@emotion/react';
import { css } from '@emotion/react';
import { useMemo } from 'react';

export const METRIC_CHART_SECTION_HEADER_SIZE = 55;

interface MetricChartsAccordionProps {
  activeKey?: string | string[];
  onActiveKeyChange?: (key: string | string[]) => void;
  children: React.ReactNode;
  disableCollapse?: boolean;
}

const MetricChartsAccordion = ({
  activeKey,
  onActiveKeyChange,
  children,
  disableCollapse = false,
}: MetricChartsAccordionProps) => {
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
        display: disableCollapse ? 'none' : undefined,
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
        height: METRIC_CHART_SECTION_HEADER_SIZE,
      },
    };
    return styles;
  }, [theme, clsPrefix, disableCollapse]);

  return (
    <Accordion
      componentId="codegen_mlflow_app_src_experiment-tracking_components_metricchartsaccordion.tsx_82"
      {...(activeKey ? { activeKey } : {})}
      {...(onActiveKeyChange ? { onChange: onActiveKeyChange } : {})}
      dangerouslyAppendEmotionCSS={css(styles)}
      dangerouslySetAntdProps={{ expandIconPosition: 'left' }}
    >
      {children}
    </Accordion>
  );
};

export default MetricChartsAccordion;
