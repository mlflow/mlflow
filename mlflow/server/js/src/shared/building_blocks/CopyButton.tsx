/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, Tooltip } from '@databricks/design-system';

type Props = {
  copyText: string;
  showLabel?: React.ReactNode;
  icon?: React.ReactNode;
};

export const CopyButton = ({ copyText, showLabel = true, icon }: Props) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const handleClick = () => {
    navigator.clipboard.writeText(copyText);
    setShowTooltip(true);
    setTimeout(() => {
      setShowTooltip(false);
    }, 3000);
  };

  const handleMouseLeave = () => {
    setShowTooltip(false);
  };

  return (
    <Tooltip
      title={
        <FormattedMessage
          defaultMessage='Copied'
          description='Tooltip text shown when copy operation completes'
        />
      }
      dangerouslySetAntdProps={{
        visible: showTooltip,
      }}
    >
      <Button
        type='primary'
        onClick={handleClick}
        onMouseLeave={handleMouseLeave}
        icon={icon}
        css={{ 'z-index': 1 }}
      >
        {showLabel && (
          <FormattedMessage defaultMessage='Copy' description='Button text for copy button' />
        )}
      </Button>
    </Tooltip>
  );
};
