import { McpIcon } from '@databricks/design-system';
import { RegistryIcon, resolveIconSrc } from '../../common/components/RegistryIcon';
import type { MCPIcon as MCPIconType } from '../types';

export { resolveIconSrc };

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
  return <RegistryIcon icons={icons} fallbackIcons={fallbackIcons} name={name} css={cssProp} DefaultGlyph={McpIcon} />;
};
