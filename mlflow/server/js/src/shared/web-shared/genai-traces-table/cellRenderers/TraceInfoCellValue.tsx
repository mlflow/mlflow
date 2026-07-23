import { Typography } from '@databricks/design-system';

import { isValidHttpUrl } from '../utils/DisplayUtils';

const ELLIPSIS_CSS = {
  display: 'block',
  overflow: 'hidden',
  whiteSpace: 'nowrap',
  textOverflow: 'ellipsis',
} as const;

/**
 * Renders a trace-info cell value (attribute / tag / custom metadata). When the
 * value is an http(s) URL it is rendered as a clickable hyperlink that opens in
 * a new tab (e.g. a link to a custom HTML rendering of a trace uploaded as an
 * artifact); otherwise it falls back to the plain ellipsized text used
 * elsewhere in the traces table.
 */
export const TraceInfoCellValue = ({ value }: { value: string }) => {
  if (isValidHttpUrl(value)) {
    return (
      <Typography.Link
        componentId="mlflow.genai_traces_table.trace_info_cell_url_link"
        href={value}
        title={value}
        css={ELLIPSIS_CSS}
        openInNewTab
      >
        {value}
      </Typography.Link>
    );
  }

  return (
    <div title={value} css={ELLIPSIS_CSS}>
      {value}
    </div>
  );
};
