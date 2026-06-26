import React, { useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { GlobeIcon, SimpleSelect, SimpleSelectOption, useDesignSystemTheme } from '@databricks/design-system';
import { I18nUtils, SUPPORTED_LOCALES } from './I18nUtils';

type LocaleSelectChangeEvent = {
  target: {
    value: string;
  };
};

export const LocaleSelector = () => {
  const { theme } = useDesignSystemTheme();
  const currentLocale = I18nUtils.getCurrentLocale();
  const localeSelectorLabelId = 'mlflow.locale_selector_label';

  const handleLocaleChange = useCallback((event: LocaleSelectChangeEvent) => {
    const nextLocale = I18nUtils.setCurrentLocale(event.target.value);
    const url = new URL(window.location.href);
    url.searchParams.set('l', nextLocale);
    window.location.assign(url.toString());
  }, []);

  const localeSelectorLabel = (
    <FormattedMessage defaultMessage="Language" description="Label for the UI language selector" />
  );

  return (
    <div
      css={{
        alignItems: 'center',
        backgroundColor: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        boxShadow: theme.shadows.sm,
        display: 'flex',
        flexWrap: 'nowrap',
        fontSize: theme.typography.fontSizeSm,
        gap: theme.spacing.xs,
        minHeight: theme.general.heightSm,
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
      }}
    >
      <GlobeIcon css={{ color: theme.colors.textSecondary, flexShrink: 0, fontSize: 16 }} />
      <span
        id={localeSelectorLabelId}
        css={{
          color: theme.colors.textSecondary,
          lineHeight: `${theme.general.heightSm}px`,
          whiteSpace: 'nowrap',
        }}
      >
        {localeSelectorLabel}
      </span>
      <SimpleSelect
        id="mlflow.locale_selector"
        componentId="mlflow.locale_selector"
        aria-labelledby={localeSelectorLabelId}
        onChange={handleLocaleChange}
        value={currentLocale}
        css={{
          minWidth: 156,
        }}
      >
        {SUPPORTED_LOCALES.map(({ locale, label }) => (
          <SimpleSelectOption key={locale} value={locale}>
            {label}
          </SimpleSelectOption>
        ))}
      </SimpleSelect>
    </div>
  );
};
