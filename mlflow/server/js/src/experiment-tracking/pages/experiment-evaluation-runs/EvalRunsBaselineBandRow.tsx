import { TableCell, TableRow, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { EvalRunsBaselineTagValue } from './EvalRunsBaseline.utils';

/**
 * The `BASELINE` section label that sits directly above the pinned baseline row.
 *
 * Pinning the row alone reads as a sort bug: sort by correctness and the top row
 * can show a lower value than the one beneath it. Labelling the band makes the
 * table read as two lists — "the baseline", then "everything else, sorted" —
 * which is what it actually is.
 *
 * This band is the *only* place the table says "baseline". The run-name cell used
 * to carry a BASELINE pill as well, which said the same word 40px below this
 * label while spending run-name width the names cannot spare. The pill is gone;
 * its tooltip — who set the baseline, and why the row survives the filter — moved
 * here, so there is one baseline marker and one place to hover for the detail.
 */
export const EvalRunsBaselineBandRow = ({
  isOutsideFilter = false,
  baseline,
  columnsWidth,
}: {
  isOutsideFilter?: boolean;
  baseline?: EvalRunsBaselineTagValue;
  /**
   * Total width of the visible columns. The band holds a single cell, so its own
   * max-content is just the label — it cannot discover the table's width itself,
   * and without this the tint stopped at the viewport edge while the data rows
   * carried on to the right.
   */
  columnsWidth?: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const provenance = baseline?.setBy
    ? intl.formatMessage(
        {
          defaultMessage: 'Baseline, set by {setBy}',
          description: 'Tooltip explaining who chose the shared baseline run for this experiment',
        },
        { setBy: baseline.setBy },
      )
    : intl.formatMessage({
        defaultMessage: 'Baseline — shared by everyone viewing this experiment',
        description: 'Tooltip noting that the baseline run is stored per-experiment, not per-user',
      });

  // Spelled out on hover, because the label only has room for the fact and not
  // the reason. Without this the row looks like the filter is broken.
  const outsideFilterNote = isOutsideFilter
    ? intl.formatMessage({
        defaultMessage:
          'This run does not match the current filters, but stays visible because every delta is measured against it.',
        description: 'Tooltip explaining why the baseline run is shown even though the filters exclude it',
      })
    : undefined;

  return (
    <TableRow
      css={{
        minHeight: 0,
        backgroundColor: theme.colors.tableBackgroundSelectedDefault,
        // Matched to the columns rather than left to flex: a single-cell row's
        // max-content is only its label, so the tint used to stop at the
        // viewport edge and the band came apart from the row it labels once the
        // table scrolled right.
        ...(columnsWidth ? { width: columnsWidth, minWidth: columnsWidth } : { minWidth: '100%' }),
      }}
    >
      <TableCell
        css={{
          paddingTop: theme.spacing.xs,
          paddingBottom: 0,
          fontSize: 9,
          fontWeight: 600,
          letterSpacing: '0.06em',
          color: theme.colors.textValidationInfo,
          whiteSpace: 'nowrap',
        }}
      >
        <Tooltip
          componentId="mlflow.eval-runs.baseline-band.tooltip"
          content={outsideFilterNote ? `${provenance}. ${outsideFilterNote}` : provenance}
        >
          {/*
            The qualifier is in the always-visible label rather than only in the
            tooltip: a row that contradicts the active filter is exactly the case
            that must not need a hover to explain itself.
          */}
          {/*
            No tabIndex: the tooltip is enrichment (who set the baseline), and the
            one fact a keyboard user must not miss — that this row defies the
            active filter — is in the label text itself, not the hover.
          */}
          <span>
            {isOutsideFilter ? (
              <FormattedMessage
                defaultMessage="BASELINE · NOT IN CURRENT FILTER"
                description="Section label above the pinned baseline run when the active filters exclude that run"
              />
            ) : (
              <FormattedMessage
                defaultMessage="BASELINE"
                description="Section label above the pinned baseline run in the evaluation runs table"
              />
            )}
          </span>
        </Tooltip>
      </TableCell>
    </TableRow>
  );
};
