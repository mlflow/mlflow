import { cva, type VariantProps, cx } from 'class-variance-authority';
import { useColorMode } from '@docusaurus/theme-common';

import LogoDark from '@site/static/images/mlflow-logo-white.svg';
import LogoLight from '@site/static/images/mlflow-logo-black.svg';
import BlueBg from '@site/static/images/footer-blue-bg.png';
import RedBg from '@site/static/images/footer-red-bg.png';
import ColorfulBg from '@site/static/images/footer-colorful-bg.png';
import BlueBgLight from '@site/static/images/footer-blue-bg-light.png';
import RedBgLight from '@site/static/images/footer-red-bg-light.png';
import ColorfulBgLight from '@site/static/images/footer-colorful-bg-light.png';
import useBaseUrl from '@docusaurus/useBaseUrl';

import { FooterMenuItem } from '../FooterMenuItem/FooterMenuItem';

const footerVariants = cva(
  'pb-30 flex flex-col pt-30 bg-bottom bg-no-repeat bg-cover bg-size-[auto_360px] 2xl:bg-size-[100%_360px]',
  {
    variants: {
      theme: {
        dark: 'bg-brand-black',
        light: 'bg-white',
      },
    },
  },
);

const getBackgroundImage = (variant: 'blue' | 'red' | 'colorful', isDarkMode: boolean) => {
  if (isDarkMode) {
    switch (variant) {
      case 'blue':
        return BlueBg;
      case 'red':
        return RedBg;
      case 'colorful':
        return ColorfulBg;
    }
  } else {
    switch (variant) {
      case 'blue':
        return BlueBgLight;
      case 'red':
        return RedBgLight;
      case 'colorful':
        return ColorfulBgLight;
    }
  }
};

const getLogo = (isDarkMode: boolean) => {
  return isDarkMode ? LogoDark : LogoLight;
};

const getBackgroundGradient = (isDarkMode: boolean) => {
  if (isDarkMode) {
    return `linear-gradient(#0e1414 60%, #0c141400 90% 100%)`;
  } else {
    return `linear-gradient(#ffffff 60%, #ffffff00 90% 100%)`;
  }
};

export const Footer = ({ variant }: { variant: 'blue' | 'red' | 'colorful' }) => {
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const Logo = getLogo(isDarkMode);
  const backgroundGradient = getBackgroundGradient(isDarkMode);

  return (
    <footer
      className={cx(footerVariants({ theme: colorMode }))}
      style={{ backgroundImage: `${backgroundGradient},url(${getBackgroundImage(variant, isDarkMode)})` }}
    >
      <div className="flex flex-row justify-between items-start md:items-center px-6 lg:px-20 gap-10 xs:gap-0 max-w-container">
        <div className="flex flex-col gap-8">
          <Logo className="h-[36px] shrink-0" />
          <div className="text-xs text-left md:text-nowrap md:w-0">
            © 2025 MLflow Project, a Series of LF Projects, LLC.
          </div>
        </div>

        <div className="flex flex-col flex-wrap justify-end md:text-right md:flex-row gap-x-10 lg:gap-x-20 gap-y-5 w-2/5 md:w-auto md:pt-2 max-w-fit">
          {/* these routes are on the main mlflow.org site, which is hosted in a different repo */}
          <FooterMenuItem href="https://mlflow.org" isDarkMode={isDarkMode}>
            Components
          </FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/releases" isDarkMode={isDarkMode}>
            Releases
          </FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/blog" isDarkMode={isDarkMode}>
            Blog
          </FooterMenuItem>
          <FooterMenuItem href={useBaseUrl('/')} isDarkMode={isDarkMode}>
            Docs
          </FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/ambassadors" isDarkMode={isDarkMode}>
            Ambassador Program
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};
