/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useState } from 'react';
import { Button } from '@databricks/design-system';

type Props = {
  text: string;
  maxSize: number;
  className: string;
  allowShowMore?: boolean;
  dataTestId?: string;
};

export const TrimmedText = ({ text, maxSize, className, allowShowMore = false, dataTestId }: Props) => {
  if (text.length <= maxSize) {
    return (
      <span className={className} data-testid={dataTestId}>
        {text}
      </span>
    );
  }
  const trimmedText = `${text.substr(0, maxSize)}...`;
  // Reported during ESLint upgrade
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [showMore, setShowMore] = useState(false);
  return (
    <span className={className} data-testid={dataTestId}>
      {showMore ? text : trimmedText}
      {allowShowMore && (
        <Button
          componentId="codegen_mlflow_app_src_common_components_trimmedtext.tsx_30"
          type="link"
          onClick={() => setShowMore(!showMore)}
          size="small"
          css={styles.expandButton}
          data-test-id="trimmed-text-button"
        >
          {showMore ? 'collapse' : 'expand'}
        </Button>
      )}
    </span>
  );
};

const styles = {
  expandButton: {
    display: 'inline-block',
  },
};
