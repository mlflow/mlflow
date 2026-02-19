import { useLocation } from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { Footer } from '@site/src/components/Footer/Footer';
import React from 'react';

function getFooterVariant(pathname: string) {
  const genAI = useBaseUrl('/genai');
  const classicalML = useBaseUrl('/ml');

  if (pathname.startsWith(genAI)) {
    return 'red';
  } else if (pathname.startsWith(classicalML)) {
    return 'blue';
  } else {
    return 'colorful';
  }
}

function FooterWrapper() {
  const location = useLocation();
  const variant = getFooterVariant(location.pathname);
  return <Footer variant={variant} />;
}

export default React.memo(FooterWrapper);
