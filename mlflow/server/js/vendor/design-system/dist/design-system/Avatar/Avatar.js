import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useMemo } from 'react';
import { SvgDatabricksIcon } from './SvgDatabricksIcon';
import { SvgUserIcon } from './SvgUserIcon';
import { useDesignSystemTheme } from '../Hooks';
import { SparkleDoubleIcon } from '../Icon';
import { OverflowPopover } from '../Overflow/OverflowPopover';
const SIZE = new Map([
    ['xl', { avatarSize: 48, fontSize: 18, groupShift: 12, iconSize: 24 }],
    ['lg', { avatarSize: 40, fontSize: 16, groupShift: 8, iconSize: 20 }],
    ['md', { avatarSize: 32, fontSize: 14, groupShift: 4, iconSize: 16 }],
    ['sm', { avatarSize: 24, fontSize: 12, groupShift: 4, iconSize: 14 }],
    ['xs', { avatarSize: 20, fontSize: 12, groupShift: 2, iconSize: 12 }],
    ['xxs', { avatarSize: 16, fontSize: 11, groupShift: 2, iconSize: 12 }],
]);
const DEFAULT_SIZE = 'sm';
function getAvatarEmotionStyles({ backgroundColor, size = DEFAULT_SIZE, theme, }) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
    const { avatarSize, fontSize, iconSize } = SIZE.get(size);
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
        },
        system: {
            borderRadius: theme.borders.borderRadiusSm,
            overflow: 'hidden',
        },
        user: {
            borderRadius: '100%',
            overflow: 'hidden',
        },
        userIcon: {
            alignItems: 'flex-end',
        },
    };
}
/** Generate random number from a string between 0 - (maxRange - 1) */
function getRandomNumberFromString({ value, maxRange }) {
    let hash = 0;
    let char = 0;
    if (value.length === 0)
        return hash;
    for (let i = 0; i < value.length; i++) {
        char = value.charCodeAt(i);
        hash = (hash << 5) - hash + char;
        hash = hash & hash;
    }
    const idx = Math.abs(hash % maxRange);
    return idx;
}
function getAvatarBackgroundColor(label, theme) {
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
export function Avatar(props) {
    const { theme } = useDesignSystemTheme();
    const styles = getAvatarEmotionStyles({
        size: props.size,
        theme,
        backgroundColor: 'backgroundColor' in props ? props.backgroundColor : undefined,
    });
    switch (props.type) {
        case 'entity':
            if ('src' in props && props.src) {
                return _jsx("img", { css: [styles.default, styles.img, styles.system], src: props.src, alt: props.label });
            }
            if ('icon' in props && props.icon) {
                return (_jsx("div", { css: [styles.default, styles.system, styles.icon], role: "img", "aria-label": props.label, children: props.icon }));
            }
            // display first initial of name when no image / icon is provided
            return (_jsx("div", { css: [
                    styles.default,
                    styles.system,
                    styles.icon,
                    { backgroundColor: getAvatarBackgroundColor(props.label, theme) },
                ], children: _jsx("abbr", { css: styles.abbr, title: props.label, children: props.label.substring(0, 1) }) }));
        case 'user':
            if ('label' in props && props.label.trim()) {
                if (props.src) {
                    return _jsx("img", { css: [styles.default, styles.img, styles.user], src: props.src, alt: props.label });
                }
                else if (props.icon) {
                    return (_jsx("div", { css: [styles.default, styles.user, styles.icon], role: "img", "aria-label": props.label, children: props.icon }));
                }
                // display first initial of name when no image / icon is provided
                return (_jsx("div", { css: [
                        styles.default,
                        styles.user,
                        styles.icon,
                        { backgroundColor: getAvatarBackgroundColor(props.label, theme) },
                    ], children: _jsx("abbr", { css: styles.abbr, title: props.label, children: props.label.substring(0, 1) }) }));
            }
            // default to user icon when no user info is provided
            return (_jsx("div", { css: [styles.default, styles.user, styles.icon, styles.userIcon], role: "img", "aria-label": "user", children: _jsx(SvgUserIcon, {}) }));
    }
}
export function DBAssistantAvatar({ size }) {
    return _jsx(Avatar, { size: size, type: "entity", label: "Assistant", icon: _jsx(SvgDatabricksIcon, {}) });
}
export function AssistantAvatar({ backgroundColor, size, }) {
    return (_jsx(Avatar, { backgroundColor: backgroundColor, size: size, type: "entity", label: "Assistant", icon: _jsx(SparkleDoubleIcon, {}) }));
}
const MAX_AVATAR_GROUP_USERS = 3;
function getAvatarGroupEmotionStyles(theme) {
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
        },
    };
}
export function AvatarGroup({ size = DEFAULT_SIZE, users }) {
    const { theme } = useDesignSystemTheme();
    const styles = getAvatarGroupEmotionStyles(theme);
    const displayedUsers = useMemo(() => users.slice(0, MAX_AVATAR_GROUP_USERS), [users]);
    const extraUsers = useMemo(() => users.slice(MAX_AVATAR_GROUP_USERS), [users]);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
    const { avatarSize, groupShift: avatarShift, fontSize } = SIZE.get(size);
    return (_jsxs("div", { css: styles.container, children: [_jsx("div", { css: {
                    ...styles.avatarsContainer,
                    width: (avatarSize + 2 - avatarShift) * displayedUsers.length + avatarShift,
                }, children: displayedUsers.map((user, idx) => (_jsx("div", { css: { ...styles.avatar, left: -avatarShift * idx }, children: _jsx(Avatar, { size: size, type: "user", ...user }) }, `${user.label}-idx`))) }), extraUsers.length > 0 && (_jsx(OverflowPopover, { items: extraUsers.map((user) => user.label), tooltipText: "Show more users", renderLabel: (label) => _jsx("span", { css: { fontSize: `${fontSize}px !important` }, children: label }) }))] }));
}
//# sourceMappingURL=Avatar.js.map