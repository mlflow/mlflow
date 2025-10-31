import { useLocation } from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { Footer } from '@site/src/components/Footer/Footer';
import React from 'react';
function getFooterVariant(pathname) {
    var genAI = useBaseUrl('/genai');
    var classicalML = useBaseUrl('/ml');
    if (pathname.startsWith(genAI)) {
        return 'red';
    }
    else if (pathname.startsWith(classicalML)) {
        return 'blue';
    }
    else {
        return 'colorful';
    }
}
function FooterWrapper() {
    var location = useLocation();
    var variant = getFooterVariant(location.pathname);
    return <Footer variant={variant}/>;
}
export default React.memo(FooterWrapper);
