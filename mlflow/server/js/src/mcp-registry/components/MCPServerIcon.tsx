import { McpIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { MCPIcon as MCPIconType } from '../types';
import { mcpIconStyles } from '../styles';
import { resolveIcon, sanitizeHref } from '../utils';
import { useIconFallback } from '../hooks/useIconFallback';

const ICON_SIZE = 16;

export const resolveIconSrc = (
  icons?: MCPIconType[],
  fallbackIcons?: MCPIconType[],
  isDarkMode?: boolean,
): string | undefined => resolveIcon(icons, isDarkMode)?.src ?? resolveIcon(fallbackIcons, isDarkMode)?.src;

export const MCPServerIcon = ({
  icons,
  fallbackIcons,
  name,
  css: cssProp,
}: {
  icons?: MCPIconType[];
  fallbackIcons?: MCPIconType[];
  name?: string;
  css?: Record<string, unknown>;
}) => {
  const { theme } = useDesignSystemTheme();
  const primarySrc = sanitizeHref(resolveIcon(icons, theme.isDarkMode)?.src);
  const fallbackSrc = sanitizeHref(resolveIcon(fallbackIcons, theme.isDarkMode)?.src);
  const { activeSrc, onError } = useIconFallback(primarySrc, fallbackSrc);

  if (activeSrc) {
    return (
      <img
        src={activeSrc}
        alt={name || ''}
        referrerPolicy="no-referrer"
        onError={onError}
        css={{
          width: ICON_SIZE,
          height: ICON_SIZE,
          objectFit: 'contain',
          ...mcpIconStyles(theme),
          ...cssProp,
        }}
      />
    );
  }

  return <McpIcon aria-hidden css={{ width: ICON_SIZE, height: ICON_SIZE, ...mcpIconStyles(theme), ...cssProp }} />;
};
