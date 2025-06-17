import React from 'react';
import { useLocation } from '@docusaurus/router';
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem';
import styles from './DocsDropdown.module.css';

type Section = 'genai' | 'ml' | 'default';

interface DocsDropdownProps {
  mobile?: boolean;
  position?: 'left' | 'right';
  items?: any[];
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
    if (path.includes('/genai') || path.startsWith('/genai')) {
      return 'genai';
    } else if (path.includes('/ml') || path.startsWith('/ml')) {
      return 'ml';
    }
    return 'default';
  };

  const currentSection = getCurrentSection();

  const getLabel = (): string => {
    switch (currentSection) {
      case 'genai':
        return 'GenAI Docs';
      case 'ml':
        return 'ML Docs';
      default:
        return configLabel || 'Documentation';
    }
  };

  const dropdownItems = configItems || [
    {
      type: 'docSidebar',
      sidebarId: 'classicMLSidebar',
      label: '🤖 ML Documentation',
      docsPluginId: 'classic-ml',
      className: styles.mlDocsLink,
    },
    {
      type: 'docSidebar',
      sidebarId: 'genAISidebar',
      label: '🧠 GenAI Documentation',
      docsPluginId: 'genai',
      className: styles.genaiDocsLink,
    },
  ];

  const getDropdownClassName = (): string => {
    const baseClass = 'docs-dropdown';
    const sectionClass = currentSection !== 'default' ? `docs-dropdown-${currentSection}` : '';
    return `${baseClass} ${sectionClass}`.trim();
  };

  return (
    <DropdownNavbarItem
      {...props}
      mobile={mobile}
      label={getLabel()}
      items={dropdownItems}
      className={getDropdownClassName()}
      data-active={currentSection !== 'default' ? currentSection : undefined}
    />
  );
}