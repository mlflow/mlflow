import { type Theme, type CSSObject } from '@emotion/react';
import React, { useMemo } from 'react';

import { SvgDatabricksIcon } from './SvgDatabricksIcon';
import { SvgUserIcon } from './SvgUserIcon';
import { useDesignSystemTheme } from '../Hooks';
import { SparkleDoubleIcon } from '../Icon';
import { OverflowPopover } from '../Overflow/OverflowPopover';

export type AvatarSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type AvatarBackgroundColor = 'indigo' | 'teal' | 'pink' | 'purple' | 'brown';
export type AvatarType = 'user' | 'entity';

interface AvatarBasicProps {
  /** Type of the avatar: user or entity */
  type: AvatarType;
  /** The label is used to reinforce the Avatar and is most likely the name of the user or entity. */
  label: string;
  /** Size of the avatar */
  size?: AvatarSize;
}

interface AvatarImgProps extends AvatarBasicProps {
  /** Url for entity image */
  src?: string;
  /** Icon for entity image */
  icon?: never;
}

interface AvatarIconProps extends AvatarBasicProps {
  /** Url for entity image */
  src?: never;
  /** Icon for entity image */
  icon?: React.ReactNode;
  /** Background color of the avatar. Only applicable for icon avatar */
  backgroundColor?: AvatarBackgroundColor;
}

// For user, either src or icon can be defined or none of both.
// For entity, either src or icon must be defined.
export type AvatarProps =
  | {
      /** Type of the avatar: user or entity */
      type: 'user';
      /** Size of the avatar */
      size?: AvatarSize;
    }
  | AvatarImgProps
  | AvatarIconProps;

const SIZE = new Map<AvatarSize, { avatarSize: number; fontSize: number; groupShift: number; iconSize: number }>([
  ['xl', { avatarSize: 48, fontSize: 18, groupShift: 12, iconSize: 24 }],
  ['lg', { avatarSize: 40, fontSize: 16, groupShift: 8, iconSize: 20 }],
  ['md', { avatarSize: 32, fontSize: 14, groupShift: 4, iconSize: 16 }],
  ['sm', { avatarSize: 24, fontSize: 12, groupShift: 4, iconSize: 14 }],
  ['xs', { avatarSize: 20, fontSize: 12, groupShift: 2, iconSize: 12 }],
]);
const DEFAULT_SIZE = 'sm';

function getAvatarEmotionStyles({
  backgroundColor,
  size = DEFAULT_SIZE,
  theme,
}: {
  backgroundColor?: AvatarBackgroundColor;
  size?: AvatarSize;
  theme: Theme;
}) {
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
  const { avatarSize, fontSize, iconSize } = SIZE.get(size)!;

  return {
    abbr: {
      color: theme.colors.tagText,
      textDecoration: 'none',
      textTransform: 'uppercase',
    },
    default: {
      height: avatarSize,
      width: avatarSize,
      fontSize,
      [`.${theme.general.iconfontCssPrefix}`]: {
        fontSize: iconSize,
      },
    },
    icon: {
      alignItems: 'center',
      color: backgroundColor ? theme.colors.tagText : theme.colors.textSecondary,
      backgroundColor: backgroundColor ? theme.colors[backgroundColor] : theme.colors.tagDefault,
      display: 'flex',
      justifyContent: 'center',
    },
    img: {
      objectFit: 'cover',
      objectPosition: 'center',
    } as CSSObject,
    system: {
      borderRadius: theme.legacyBorders.borderRadiusMd,
      overflow: 'hidden',
    },
    user: {
      borderRadius: '100%',
      overflow: 'hidden',
    },
    userIcon: {
      alignItems: 'flex-end',
    },
  } satisfies Record<string, CSSObject>;
}

/** Generate random number from a string between 0 - (maxRange - 1) */
function getRandomNumberFromString({ value, maxRange }: { value: string; maxRange: number }) {
  let hash = 0;
  let char = 0;

  if (value.length === 0) return hash;

  for (let i = 0; i < value.length; i++) {
    char = value.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }

  const idx = Math.abs(hash % maxRange);

  return idx;
}

function getAvatarBackgroundColor(label: string, theme: Theme) {
  const randomNumber = getRandomNumberFromString({ value: label, maxRange: 5 });

  switch (randomNumber) {
    case 0:
      return theme.colors.indigo;
    case 1:
      return theme.colors.teal;
    case 2:
      return theme.colors.pink;
    case 3:
      return theme.colors.brown;
    case 4:
    default:
      return theme.colors.purple;
  }
}

export function Avatar(props: AvatarProps) {
  const { theme } = useDesignSystemTheme();
  const styles = getAvatarEmotionStyles({
    size: props.size,
    theme,
    backgroundColor: 'backgroundColor' in props ? props.backgroundColor : undefined,
  });

  switch (props.type) {
    case 'entity':
      if ('src' in props && props.src) {
        return <img css={[styles.default, styles.img, styles.system]} src={props.src} alt={props.label} />;
      }

      if ('icon' in props && props.icon) {
        return (
          <div css={[styles.default, styles.system, styles.icon]} role="img" aria-label={props.label}>
            {props.icon}
          </div>
        );
      }

      // display first initial of name when no image / icon is provided
      return (
        <div
          css={[
            styles.default,
            styles.system,
            styles.icon,
            { backgroundColor: getAvatarBackgroundColor(props.label, theme) },
          ]}
        >
          <abbr css={styles.abbr} title={props.label}>
            {props.label.substring(0, 1)}
          </abbr>
        </div>
      );

    case 'user':
      if ('label' in props && props.label.trim()) {
        if (props.src) {
          return <img css={[styles.default, styles.img, styles.user]} src={props.src} alt={props.label} />;
        } else if (props.icon) {
          return (
            <div css={[styles.default, styles.user, styles.icon]} role="img" aria-label={props.label}>
              {props.icon}
            </div>
          );
        }
        // display first initial of name when no image / icon is provided
        return (
          <div
            css={[
              styles.default,
              styles.user,
              styles.icon,
              { backgroundColor: getAvatarBackgroundColor(props.label, theme) },
            ]}
          >
            <abbr css={styles.abbr} title={props.label}>
              {props.label.substring(0, 1)}
            </abbr>
          </div>
        );
      }

      // default to user icon when no user info is provided
      return (
        <div css={[styles.default, styles.user, styles.icon, styles.userIcon]} role="img" aria-label="user">
          <SvgUserIcon />
        </div>
      );
  }
}

export function DBAssistantAvatar({ size }: { size?: AvatarSize }) {
  return <Avatar size={size} type="entity" label="Assistant" icon={<SvgDatabricksIcon />} />;
}

export function AssistantAvatar({
  backgroundColor,
  size,
}: {
  backgroundColor?: AvatarBackgroundColor;
  size?: AvatarSize;
}) {
  return (
    <Avatar
      backgroundColor={backgroundColor}
      size={size}
      type="entity"
      label="Assistant"
      icon={<SparkleDoubleIcon />}
    />
  );
}

const MAX_AVATAR_GROUP_USERS = 3;

export interface AvatarGroupProps {
  size?: AvatarSize;
  users: (Omit<AvatarImgProps, 'type'> | Omit<AvatarIconProps, 'type'>)[];
}

function getAvatarGroupEmotionStyles(theme: Theme) {
  return {
    container: {
      display: 'flex',
      alignItems: 'center',
      gap: theme.spacing.xs,
    },
    avatarsContainer: {
      display: 'flex',
    },
    avatar: {
      display: 'flex',
      borderRadius: '100%',
      border: `1px solid ${theme.colors.backgroundPrimary}`,
      position: 'relative',
    } as CSSObject,
  };
}

export function AvatarGroup({ size = DEFAULT_SIZE, users }: AvatarGroupProps) {
  const { theme } = useDesignSystemTheme();
  const styles = getAvatarGroupEmotionStyles(theme);
  const displayedUsers = useMemo(() => users.slice(0, MAX_AVATAR_GROUP_USERS), [users]);
  const extraUsers = useMemo(() => users.slice(MAX_AVATAR_GROUP_USERS), [users]);
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
  const { avatarSize, groupShift: avatarShift, fontSize } = SIZE.get(size)!;

  return (
    <div css={styles.container}>
      <div
        css={{
          ...styles.avatarsContainer,
          width: (avatarSize + 2 - avatarShift) * displayedUsers.length + avatarShift,
        }}
      >
        {displayedUsers.map((user, idx) => (
          <div css={{ ...styles.avatar, left: -avatarShift * idx }} key={`${user.label}-idx`}>
            <Avatar size={size} type="user" {...user} />
          </div>
        ))}
      </div>
      {extraUsers.length > 0 && (
        <OverflowPopover
          items={extraUsers.map((user) => user.label)}
          tooltipText="Show more users"
          renderLabel={(label) => <span css={{ fontSize: `${fontSize}px !important` }}>{label}</span>}
        />
      )}
    </div>
  );
}
