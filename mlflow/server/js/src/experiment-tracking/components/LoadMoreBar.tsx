/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/**
 "Load more" bar for user to click and load more runs. This row is currently built
 outside of the Table component as we are following a minimum-invasive way of building
 this feature to avoid massive refactor on current implementation. Ideally, this row
 can be built inside the Table as a special row by rewriting table rendering with a
 custom `rowRenderer`. That way, we don't need to handle scrolling position manually.
 We can consider doing this refactor while we implement the multi-level nested runs.
 TODO(Zangr) rewrite the runs table with rowRenderer to allow a built-in load-more row
*/

import React from 'react';
import { Tooltip, Button, WithDesignSystemThemeHoc, SyncIcon } from '@databricks/design-system';
import { injectIntl, FormattedMessage } from 'react-intl';

type Props = {
  style?: any;
  loadingMore: boolean;
  onLoadMore: (...args: any[]) => any;
  disableButton?: boolean;
  nestChildren?: boolean;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
  designSystemThemeApi: any;
};

export class LoadMoreBarImpl extends React.PureComponent<Props> {
  renderButton() {
    const { disableButton, onLoadMore, nestChildren, intl, designSystemThemeApi } = this.props;
    const loadMoreButton = (
      <Button
        componentId="codegen_mlflow_app_src_experiment-tracking_components_loadmorebar.tsx_38"
        className="load-more-button"
        style={styles.loadMoreButton}
        type="primary"
        htmlType="button"
        onClick={onLoadMore}
        size="small"
        disabled={disableButton}
      >
        <FormattedMessage defaultMessage="Load more" description="Load more button text to load more experiment runs" />
      </Button>
    );

    if (disableButton) {
      return (
        <Tooltip
          // @ts-expect-error TS(2322): Type '{ children: Element; className: string; plac... Remove this comment to see the full error message
          className="load-more-button-disabled-tooltip"
          placement="bottom"
          title={intl.formatMessage({
            defaultMessage: 'No more runs to load.',
            description: 'Tooltip text for load more button when there are no more experiment runs to load',
          })}
        >
          {loadMoreButton}
        </Tooltip>
      );
    } else if (nestChildren) {
      return (
        <div>
          {loadMoreButton}
          <Tooltip
            // @ts-expect-error TS(2322): Type '{ children: Element; className: string; plac... Remove this comment to see the full error message
            className="load-more-button-nested-info-tooltip"
            placement="bottom"
            title={intl.formatMessage({
              defaultMessage: 'Loaded child runs are nested under their parents.',
              description:
                // eslint-disable-next-line max-len
                'Tooltip text for load more button explaining the runs are nested under their parent experiment run',
            })}
          >
            <i
              className="fas fa-info-circle"
              css={{
                marginLeft: designSystemThemeApi.theme.spacing.sm,
                color: designSystemThemeApi.theme.colors.actionPrimaryBackgroundDefault,
              }}
            />
          </Tooltip>
        </div>
      );
    } else {
      return loadMoreButton;
    }
  }

  render() {
    const { loadingMore, style } = this.props;
    return (
      <div className="load-more-row" style={{ ...styles.loadMoreRows, ...style }}>
        {loadingMore ? (
          <div className="loading-more-wrapper" style={styles.loadingMoreWrapper}>
            <SyncIcon spin style={styles.loadingMoreIcon} />
          </div>
        ) : (
          this.renderButton()
        )}
      </div>
    );
  }
}

const styles = {
  loadMoreRows: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    background: 'white',
  },
  loadingMoreWrapper: {
    display: 'flex',
    alignItems: 'center',
  },
  loadingMoreIcon: {
    fontSize: 20,
  },
  loadMoreButton: {
    paddingLeft: 16,
    paddingRight: 16,
  },
};

// @ts-expect-error TS(2769): No overload matches this call.
export const LoadMoreBar = WithDesignSystemThemeHoc(injectIntl(LoadMoreBarImpl));
