import type { ReactNode } from 'react';
import { Alert, Button, CloseIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

/**
 * Read-only banner shown while the user is viewing a shared/saved view. It signals that the view is
 * read-only with respect to local persistence — nothing the user does is saved — and hosts the
 * explicit actions to leave it.
 *
 * Presentational and data-source-agnostic so both the runs and traces tabs can render it:
 * - Runs passes `onOverride` (adopt the shared view into the user's own saved view) + `onDiscard`.
 * - Traces omits `onOverride`, so only the Discard button renders.
 *
 * `componentId` is the base id for the analytics events; the override/discard buttons derive
 * `.override` / `.discard` from it. It is threaded from the consumer (rather than hardcoded) so each
 * tab keeps its own registered componentId namespace.
 */
// Override is all-or-nothing: a consumer that wires up `onOverride` must also supply the button's
// `overrideLabel`, otherwise the override button would render empty. Consumers with no override
// (e.g. traces) omit both. The union enforces this at the type level.
type OverrideProps =
  | { onOverride: () => void; overrideLabel: ReactNode }
  | { onOverride?: never; overrideLabel?: never };

export const SharedViewBanner = ({
  componentId,
  message,
  overrideLabel,
  onOverride,
  onDiscard,
  onDismiss,
}: {
  componentId: string;
  message: ReactNode;
  onDiscard: () => void;
  // When provided, renders a dismiss (X) control that hides the banner WITHOUT leaving the shared
  // view (unlike Discard, which reverts to the user's own view). Consumers still expose Override /
  // Discard elsewhere (the Views menu) so those actions survive dismissal.
  onDismiss?: () => void;
} & OverrideProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <Alert
      componentId={`${componentId}.banner`}
      type="info"
      closable={false}
      css={{ marginBottom: theme.spacing.sm }}
      message={
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.md,
            flexWrap: 'wrap',
          }}
        >
          <span>{message}</span>
          <div css={{ display: 'flex', gap: theme.spacing.sm, flexShrink: 0 }}>
            {onOverride && (
              <Button componentId={`${componentId}.override`} type="primary" size="small" onClick={onOverride}>
                {overrideLabel}
              </Button>
            )}
            <Button componentId={`${componentId}.discard`} size="small" onClick={onDiscard}>
              <FormattedMessage
                defaultMessage="Discard"
                description="Experiment page > shared view banner > button that discards the shared view and restores the user's own view"
              />
            </Button>
            {onDismiss && (
              <Tooltip
                componentId={`${componentId}.dismiss_tooltip`}
                content={
                  <FormattedMessage
                    defaultMessage="Dismiss this banner. You'll keep viewing the shared view."
                    description="Tooltip on the shared view banner's dismiss button explaining it hides the banner but keeps the view"
                  />
                }
              >
                <Button
                  componentId={`${componentId}.dismiss`}
                  size="small"
                  icon={<CloseIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Hide banner',
                    description:
                      'Experiment page > shared view banner > button that hides the banner while keeping the shared view applied',
                  })}
                  onClick={onDismiss}
                />
              </Tooltip>
            )}
          </div>
        </div>
      }
    />
  );
};
