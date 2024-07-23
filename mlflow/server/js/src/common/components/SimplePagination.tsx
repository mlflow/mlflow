/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Pagination } from '@databricks/design-system';

type Props = {
  currentPage: number;
  isLastPage: boolean;
  onClickNext: (...args: any[]) => any;
  onClickPrev: (...args: any[]) => any;
  maxResultOptions?: any[];
  handleSetMaxResult?: (...args: any[]) => any;
  getSelectedPerPageSelection: (...args: any[]) => any;
  removeBottomSpacing?: boolean;
};

export class SimplePagination extends React.Component<Props> {
  render() {
    const { currentPage, isLastPage, onClickNext, onClickPrev, maxResultOptions, removeBottomSpacing } = this.props;
    const numEntries = this.props.getSelectedPerPageSelection();
    let total;

    // This is necessary because for tables using this component, we do not know the total records.
    // Thus this is a proxy to determine whether or not the "next" button should be disabled ==>
    // if it's the last page, we set total = max number of entries, else we add 1 page more than
    // the current page.
    if (isLastPage) {
      total = currentPage * numEntries;
    } else {
      total = (currentPage + 1) * numEntries;
    }

    return (
      <div
        className="pagination-section"
        css={[classNames.wrapper, classNames.paginationOverride, removeBottomSpacing && classNames.removeBottomSpacing]}
      >
        {/* @ts-expect-error TS(2741): Property 'pageSize' is missing in type '{ currentP... Remove this comment to see the full error message */}
        <Pagination
          currentPageIndex={currentPage}
          numTotal={total}
          onChange={(nextPage) => {
            // Fire callbacks only if new page differs from the previous one
            if (nextPage !== currentPage) {
              nextPage > currentPage ? onClickNext() : onClickPrev();
            }
          }}
          dangerouslySetAntdProps={{
            showQuickJumper: false,
            showSizeChanger: true,
            pageSize: numEntries,
            pageSizeOptions: maxResultOptions,
            // @ts-expect-error TS(2722): Cannot invoke an object which is possibly 'undefin... Remove this comment to see the full error message
            onShowSizeChange: (current, size) => this.props.handleSetMaxResult({ key: size }),
          }}
        />
      </div>
    );
  }
}

const classNames = {
  wrapper: {
    textAlign: 'right',
    paddingBottom: 30,
  },
  removeBottomSpacing: {
    paddingBottom: 0,
  },
  paginationOverride: {
    // Hide extra page buttons
    '.du-bois-light-pagination-item:not(.du-bois-light-pagination-item-active)': {
      display: 'none',
    },
    // Hide jump buttons
    '.du-bois-light-pagination-jump-prev': {
      display: 'none',
    },
  },
};
