import React from 'react';
import { useLocation } from '@docusaurus/router';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import OriginalRoot from '@theme-original/Root';

export default function Root({ children }: { children: React.ReactNode }) {
  const location = useLocation();

  React.useEffect(() => {
    if (!ExecutionEnvironment.canUseDOM) return;

    const updateNavbarLabel = () => {
      const dropdownContainer = document.querySelector('.navbar__item.dropdown.dropdown--hoverable');
      if (!dropdownContainer) return;
      
      const dropdownLink = dropdownContainer.querySelector('.navbar__link, a[role="button"]');
      if (!dropdownLink) return;

      const path = location.pathname;
      let newLabel = 'Documentation';

      if (path.includes('/genai') || path.startsWith('/genai')) {
        newLabel = 'GenAI Docs';
        dropdownLink.setAttribute('data-active', 'genai');
        dropdownLink.classList.add('docs-dropdown-genai');
        dropdownLink.classList.remove('docs-dropdown-ml');
      } else if (path.includes('/ml') || path.startsWith('/ml')) {
        newLabel = 'ML Docs';
        dropdownLink.setAttribute('data-active', 'ml');
        dropdownLink.classList.add('docs-dropdown-ml');
        dropdownLink.classList.remove('docs-dropdown-genai');
      } else {
        dropdownLink.removeAttribute('data-active');
        dropdownLink.classList.remove('docs-dropdown-ml', 'docs-dropdown-genai');
      }

      dropdownLink.textContent = newLabel;
    };

    const timeouts = [0, 100, 500];
    const timers = timeouts.map(delay => setTimeout(updateNavbarLabel, delay));

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [location.pathname]);

  return <OriginalRoot>{children}</OriginalRoot>;
}