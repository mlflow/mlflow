import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import type {
  TableColumnGroupType as AntDTableColumnGroupType,
  TableColumnProps as AntDTableColumnProps,
  TableColumnsType as AntDTableColumnsType,
  TableColumnType as AntDTableColumnType,
  TablePaginationConfig as AntDTablePaginationConfig,
  TableProps as AntDTableProps,
} from 'antd';
import { Table as AntDTable } from 'antd';

import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { getPaginationEmotionStyles } from '../Pagination';
import { Spinner } from '../Spinner';
import { addDebugOutlineIfEnabled } from '../utils/debug';

// NOTE: This component is deprecated in favor of the new `Table` component, but to make room for the
// new version, we need to rename this one to `LegacyTable`. Currently, we're exporting
// both `Table` as an alias to the updated name here. Once all extant code is referencing `LegacyTable`,
// we'll delete the aliases.

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
export interface LegacyTableProps<RecordType> extends AntDTableProps<RecordType> {
  scrollableInFlexibleContainer?: boolean;
}

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
export interface LegacyTablePaginationConfig extends AntDTablePaginationConfig {}

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
export interface LegacyTableColumnGroupType<RecordType> extends AntDTableColumnGroupType<RecordType> {}

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
export interface LegacyTableColumnType<RecordType> extends AntDTableColumnType<RecordType> {}

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
export interface LegacyTableColumnProps<RecordType> extends AntDTableColumnProps<RecordType> {}

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
export interface LegacyTableColumnsType<RecordType> extends AntDTableColumnsType<RecordType> {}

const getTableEmotionStyles = (classNamePrefix: string, theme: Theme, scrollableInFlexibleContainer: boolean) => {
  const styles = [
    css({
      [`.${classNamePrefix}-table-pagination`]: {
        ...getPaginationEmotionStyles(classNamePrefix, theme),
      },
    }),
  ];
  if (scrollableInFlexibleContainer) {
    styles.push(getScrollableInFlexibleContainerStyles(classNamePrefix));
  }
  return styles;
};

const getScrollableInFlexibleContainerStyles = (clsPrefix: string): SerializedStyles => {
  const styles: CSSObject = {
    minHeight: 0,
    [`.${clsPrefix}-spin-nested-loading`]: { height: '100%' },
    [`.${clsPrefix}-spin-container`]: { height: '100%', display: 'flex', flexDirection: 'column' },
    [`.${clsPrefix}-table-container`]: { height: '100%', display: 'flex', flexDirection: 'column' },
    [`.${clsPrefix}-table`]: { minHeight: 0 },
    [`.${clsPrefix}-table-header`]: { flexShrink: 0 },
    [`.${clsPrefix}-table-body`]: { minHeight: 0 },
  };
  return css(styles);
};

const DEFAULT_LOADING_SPIN_PROPS = { indicator: <Spinner /> };

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
// eslint-disable-next-line @typescript-eslint/ban-types
export const LegacyTable = <T extends object>(props: LegacyTableProps<T>): JSX.Element => {
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const { loading, scrollableInFlexibleContainer, children, ...tableProps } = props;

  return (
    <DesignSystemAntDConfigProvider>
      <AntDTable
        {...addDebugOutlineIfEnabled()}
        // NOTE(FEINF-1273): The default loading indicator from AntD does not animate
        // and the design system spinner is recommended over the AntD one. Therefore,
        // if `loading` is `true`, render the design system <Spinner/> component.
        loading={loading === true ? DEFAULT_LOADING_SPIN_PROPS : loading}
        scroll={scrollableInFlexibleContainer ? { y: 'auto' } : undefined}
        {...tableProps}
        css={getTableEmotionStyles(classNamePrefix, theme, Boolean(scrollableInFlexibleContainer))}
        // ES-902549 this allows column names of "children", using a name that is less likely to be hit
        expandable={{ childrenColumnName: '__antdChildren' }}
      >
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </AntDTable>
    </DesignSystemAntDConfigProvider>
  );
};
