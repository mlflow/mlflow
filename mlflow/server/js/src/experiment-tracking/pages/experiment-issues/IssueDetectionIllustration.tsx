import { keyframes } from '@emotion/react';
import { useDesignSystemTheme } from '@databricks/design-system';

const revealBlock = keyframes({
  '0%': { opacity: 0, transform: 'translateY(25%)' },
  '7%': { opacity: 1, transform: 'translateY(0)' },
  '100%': { opacity: 1, transform: 'translateY(0)' },
});

const revealDetectionMarker = keyframes({
  '0%': { opacity: 0, transform: 'translateY(-50%) scale(0.8)' },
  '4%': { opacity: 1, transform: 'translateY(8%) scale(1.05)' },
  '7%': { opacity: 1, transform: 'translateY(0) scale(1)' },
  '100%': { opacity: 1, transform: 'translateY(0) scale(1)' },
});

const ANIMATION_DURATION_MULTIPLIER = 1.3;
const ANIMATION_CYCLE_DURATION_MULTIPLIER = 32;

interface IsometricBlock {
  x: number;
  y: number;
  isAnomaly?: boolean;
}

const ISOMETRIC_BLOCKS: IsometricBlock[] = [
  { x: 36, y: 34 },
  { x: 27, y: 39 },
  { x: 45, y: 39 },
  { x: 18, y: 44 },
  { x: 36, y: 44 },
  { x: 54, y: 44 },
  { x: 27, y: 49 },
  { x: 45, y: 49 },
  { x: 36, y: 54 },
  { x: 36, y: 19 },
  { x: 27, y: 24 },
  { x: 45, y: 24 },
  { x: 36, y: 29, isAnomaly: true },
];

const getBlockFaces = ({ x, y }: IsometricBlock) => ({
  top: `${x},${y} ${x + 9},${y + 5} ${x},${y + 10} ${x - 9},${y + 5}`,
  left: `${x - 9},${y + 5} ${x},${y + 10} ${x},${y + 17} ${x - 9},${y + 12}`,
  right: `${x + 9},${y + 5} ${x},${y + 10} ${x},${y + 17} ${x + 9},${y + 12}`,
});

export const IssueDetectionIllustration = () => {
  const { theme } = useDesignSystemTheme();
  const revealDuration = theme.animation.transitionDuration * 2 * ANIMATION_DURATION_MULTIPLIER;
  const revealDelay = Math.round((theme.animation.transitionDuration / 10) * ANIMATION_DURATION_MULTIPLIER);
  const animationCycleDuration =
    theme.animation.transitionDuration * ANIMATION_CYCLE_DURATION_MULTIPLIER * ANIMATION_DURATION_MULTIPLIER;

  return (
    <div
      css={{
        width: theme.spacing.lg * 3,
        height: theme.spacing.lg * 3,
      }}
    >
      <svg viewBox="0 0 72 72" width="100%" height="100%" focusable="false">
        {ISOMETRIC_BLOCKS.map((block, index) => {
          const faces = getBlockFaces(block);
          const outlineColor = block.isAnomaly ? theme.colors.textValidationDanger : theme.colors.actionDisabledText;
          const topColor = block.isAnomaly ? theme.colors.backgroundDanger : theme.colors.backgroundPrimary;
          const leftColor = block.isAnomaly ? theme.colors.backgroundDanger : theme.colors.backgroundSecondary;
          const rightColor = block.isAnomaly ? theme.colors.borderDanger : theme.colors.actionDisabledBackground;

          return (
            <g
              key={`${block.x}-${block.y}`}
              css={{
                transformBox: 'fill-box',
                transformOrigin: 'center',
                '@media (prefers-reduced-motion: no-preference)': {
                  animation: `${revealBlock} ${animationCycleDuration}ms ease-out infinite both`,
                  animationDelay: `${index * revealDelay}ms`,
                },
              }}
              stroke={outlineColor}
              strokeWidth="1.25"
              strokeLinejoin="round"
            >
              <polygon points={faces.top} fill={topColor} />
              <polygon points={faces.left} fill={leftColor} />
              <polygon points={faces.right} fill={rightColor} />
            </g>
          );
        })}
        <g
          css={{
            transformBox: 'fill-box',
            transformOrigin: 'center',
            '@media (prefers-reduced-motion: no-preference)': {
              animation: `${revealDetectionMarker} ${animationCycleDuration}ms ease-out infinite both`,
              animationDelay: `${revealDuration}ms`,
            },
          }}
          fill={theme.colors.backgroundPrimary}
          stroke={theme.colors.actionPrimaryBackgroundDefault}
          strokeWidth="1.5"
          strokeLinecap="round"
        >
          <circle cx="56" cy="16" r="6" />
          <path d="M60.5 20.5 65 25" />
        </g>
      </svg>
    </div>
  );
};
