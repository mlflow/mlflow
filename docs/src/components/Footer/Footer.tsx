import { cva } from 'class-variance-authority';
import { useColorMode } from '@docusaurus/theme-common';

import LogoDark from '@site/static/images/mlflow-logo-white.svg';
import LogoLight from '@site/static/images/mlflow-logo-black.svg';
import useBaseUrl from '@docusaurus/useBaseUrl';

import { FooterMenuItem } from '../FooterMenuItem/FooterMenuItem';
import { GradientWrapper, type Variant } from '../GradientWrapper/GradientWrapper';

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

const getLogo = (isDarkMode: boolean) => {
  return isDarkMode ? LogoDark : LogoLight;
};

export const Footer = ({ variant }: { variant: Variant }) => {
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const Logo = getLogo(isDarkMode);

  return (
    <GradientWrapper
      variant={variant}
      isFooter
      className={footerVariants({ theme: colorMode })}
      element="footer"
      height={360}
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
    </GradientWrapper>
  );
};
