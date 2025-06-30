import useBaseUrl from '@docusaurus/useBaseUrl';
import { useLocation } from '@docusaurus/router';
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem';
import styles from './DocsDropdown.module.css';

type Section = 'genai' | 'ml' | 'default';

interface DocsDropdownProps {
  mobile?: boolean;
  position?: 'left' | 'right';
  items: any[];
  label?: string;
  [key: string]: any;
}

export default function DocsDropdown({
  mobile,
  items: configItems,
  label: configLabel,
  ...props
}: DocsDropdownProps): JSX.Element {
  const location = useLocation();

  const getCurrentSection = (): Section => {
    const path = location.pathname;
    const genaiPath = useBaseUrl('/genai');
    const mlPath = useBaseUrl('/ml');
    if (path.startsWith(genaiPath)) {
      return 'genai';
    } else if (path.startsWith(mlPath)) {
      return 'ml';
    }
    return 'default';
  };

  const currentSection = getCurrentSection();

  const getLabel = (): JSX.Element => {
    let color;
    let text = configLabel || 'Documentation';

    if (currentSection === 'genai') {
      color = 'var(--genai-color-primary)';
      text = 'GenAI Docs';
    } else if (currentSection === 'ml') {
      color = 'var(--ml-color-primary)';
      text = 'ML Docs';
    }

    return (
      <div
        style={{
          display: 'flex',
          gap: 8,
          alignItems: 'center',
        }}
      >
        {color && (
          <div
            className={styles.dropdownCircle}
            style={{
              width: 10,
              height: 10,
              backgroundColor: color,
              borderRadius: 4,
            }}
          />
        )}
        {text}
      </div>
    );
  };

  const enhancedItems = configItems.map((item) => {
    if (item.docsPluginId === 'classic-ml') {
      return {
        ...item,
        label: (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div
              style={{
                width: 10,
                height: 10,
                backgroundColor: 'var(--ml-color-primary)',
                borderRadius: 4,
              }}
            />
            {item.label}
          </div>
        ),
      };
    } else if (item.docsPluginId === 'genai') {
      return {
        ...item,
        label: (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div
              style={{
                width: 10,
                height: 10,
                backgroundColor: 'var(--genai-color-primary)',
                borderRadius: 4,
              }}
            />
            {item.label}
          </div>
        ),
      };
    }
    return item;
  });

  return (
    <DropdownNavbarItem
      {...props}
      mobile={mobile}
      label={getLabel()}
      items={enhancedItems}
      className={styles.docsDropdown}
      data-active={currentSection !== 'default' ? currentSection : undefined}
    />
  );
}
