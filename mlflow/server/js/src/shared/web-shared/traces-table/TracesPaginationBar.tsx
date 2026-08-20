import {
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { PAGE_SIZE_OPTIONS } from './constants';
import type { PageSize } from './types';

// Module-local static analytics-id namespace (static componentId lint rule).
const COMPONENT_ID = 'web-shared.traces-table';

export interface TracesPaginationBarProps {
  pageIndex: number;
  pageSize: PageSize;
  onPageChange: (pageIndex: number) => void;
  onPageSizeChange: (pageSize: PageSize) => void;
  /** Cursor affordances derived from the token cache — drive the prev/next enabled state. */
  hasNext: boolean;
  hasPrev: boolean;
  /** Optional content pinned to the left of the bar (e.g. a result-count label), opposite the controls. */
  leadingContent?: React.ReactNode;
}

/**
 * Pagination controls: a page-size selector plus prev/next cursor buttons. The cursor search API
 * has no total, so navigation is prev/next only (Next disables on the last page via the token
 * cache's last-page marker). The row occupies a fixed height so it never pops in/out between pages.
 */
export const TracesPaginationBar: React.FC<TracesPaginationBarProps> = ({
  pageIndex,
  pageSize,
  onPageChange,
  onPageSizeChange,
  hasNext,
  hasPrev,
  leadingContent,
}: TracesPaginationBarProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: theme.spacing.md,
        minHeight: theme.general.heightSm,
      }}
    >
      {/* `marginRight: auto` pushes the rest of the row flush right, so leading content sits at the
          far left of the same bar without a wrapping flex layer. */}
      {leadingContent ? <span css={{ marginRight: 'auto' }}>{leadingContent}</span> : null}
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="Rows per page"
            description="Label for the traces table page-size selector"
          />
        </Typography.Text>
        <SimpleSelect
          componentId={`${COMPONENT_ID}.page-size`}
          id={`${COMPONENT_ID}.page-size`}
          value={String(pageSize)}
          width={80}
          aria-label={intl.formatMessage({
            defaultMessage: 'Rows per page',
            description: 'Aria label for the traces table page-size selector',
          })}
          onChange={({ target }) => onPageSizeChange(Number(target.value) as PageSize)}
        >
          {PAGE_SIZE_OPTIONS.map((size) => (
            <SimpleSelectOption key={size} value={String(size)}>
              {size}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </span>

      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Button
          componentId={`${COMPONENT_ID}.pagination.prev`}
          icon={<ChevronLeftIcon />}
          disabled={!hasPrev}
          onClick={() => onPageChange(pageIndex - 1)}
          aria-label={intl.formatMessage({
            defaultMessage: 'Previous page',
            description: 'Aria label for the previous-page button in the traces table pagination',
          })}
        />
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="Page {page}"
            description="Current page indicator in the traces table pagination"
            values={{ page: pageIndex }}
          />
        </Typography.Text>
        <Button
          componentId={`${COMPONENT_ID}.pagination.next`}
          icon={<ChevronRightIcon />}
          disabled={!hasNext}
          onClick={() => onPageChange(pageIndex + 1)}
          aria-label={intl.formatMessage({
            defaultMessage: 'Next page',
            description: 'Aria label for the next-page button in the traces table pagination',
          })}
        />
      </span>
    </div>
  );
};
