import { useState } from 'react';
import { ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';

import {
  borderedSectionContainerStyles,
  chevronContainerStyles,
  expandableRowButtonStyles,
  expandedContentPanelStyles,
} from '../styles';

export function ExpandableListSection<T>({
  items,
  getKey,
  renderRow,
  renderExpanded,
  getAriaLabel,
  footer,
}: {
  items: T[];
  getKey: (item: T, index: number) => string;
  renderRow: (ctx: { item: T; expanded: boolean }) => React.ReactNode;
  renderExpanded: (item: T) => React.ReactNode;
  getAriaLabel: (item: T, expanded: boolean) => string;
  footer?: React.ReactNode;
}) {
  const { theme } = useDesignSystemTheme();
  const [expandedKey, setExpandedKey] = useState<string | null>(null);

  return (
    <div css={borderedSectionContainerStyles(theme)}>
      {items.map((item, index) => {
        const key = getKey(item, index);
        const expanded = expandedKey === key;
        return (
          <div key={key} css={{ borderTop: index > 0 ? `1px solid ${theme.colors.border}` : 'none' }}>
            <button
              type="button"
              onClick={() => setExpandedKey(expanded ? null : key)}
              aria-expanded={expanded}
              aria-label={getAriaLabel(item, expanded)}
              css={expandableRowButtonStyles(theme)}
            >
              <div css={chevronContainerStyles(theme)}>
                {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
              </div>
              {renderRow({ item, expanded })}
            </button>
            {expanded && <div css={expandedContentPanelStyles(theme)}>{renderExpanded(item)}</div>}
          </div>
        );
      })}
      {footer}
    </div>
  );
}
