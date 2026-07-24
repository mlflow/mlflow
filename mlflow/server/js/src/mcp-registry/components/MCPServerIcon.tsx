import { useEffect, useState } from 'react';
import { McpIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { MCPIcon as MCPIconType } from '../types';
import { mcpIconStyles } from '../styles';
import { resolveIcon } from '../utils';

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
  const [imgFailed, setImgFailed] = useState(false);
  const iconSrc = resolveIconSrc(icons, fallbackIcons, theme.isDarkMode);

  useEffect(() => {
    setImgFailed(false);
  }, [iconSrc]);

  if (iconSrc && !imgFailed) {
    return (
      <img
        src={iconSrc}
        alt={name || ''}
        referrerPolicy="no-referrer"
        onError={() => setImgFailed(true)}
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
