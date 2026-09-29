import { PuzzleIcon } from '@databricks/design-system';
import { RegistryIcon } from '../../common/components/RegistryIcon';
import type { RegistryIcon as RegistryIconType } from '../types';

export const SkillIcon = ({
  icons,
  name,
  css: cssProp,
}: {
  icons?: RegistryIconType[] | null;
  name?: string;
  css?: Record<string, unknown>;
}) => {
  return <RegistryIcon icons={icons} name={name} css={cssProp} DefaultGlyph={PuzzleIcon} />;
};
