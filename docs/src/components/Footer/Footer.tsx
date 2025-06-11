import { cva, type VariantProps } from "class-variance-authority";

import Logo from "@site/static/images/mlflow-logo-white.svg";
import BlueBg from "@site/static/images/footer-blue-bg.png";
import RedBg from "@site/static/images/footer-red-bg.png";
import ColorfulBg from "@site/static/images/footer-colorful-bg.png";
import useBaseUrl from '@docusaurus/useBaseUrl';

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";

const footerVariants = cva(
  "pb-30 flex flex-col pt-30 bg-bottom bg-no-repeat bg-cover bg-size-[auto_360px] 2xl:bg-size-[100%_360px]",
);

const getBackgroundImage = (variant: "blue" | "red" | "colorful") => {
  switch (variant) {
    case "blue":
      return BlueBg;
    case "red":
      return RedBg;
    case "colorful":
      return ColorfulBg;
  }
};

export const Footer = ({ variant }: { variant: "blue" | "red" | "colorful" }) => {
  return (
    <footer className={footerVariants()} style={{ backgroundImage: `linear-gradient(#0e1414 60%,#0c141400 90% 100%),url(${getBackgroundImage(variant)})` }}>
      <div className="flex flex-row justify-between items-start px-6 lg:px-20 gap-10 xs:gap-0 max-w-container">
        <div className="flex flex-col gap-8">
          <Logo className="h-[36px] shrink-0" />
          <div className="text-xs text-[rgb(255_255_255_/_80%)] text-left md:text-nowrap md:w-0">
            © 2025 MLflow Project, a Series of LF Projects, LLC.
          </div>
        </div>

        <div className="flex flex-col flex-wrap justify-end md:text-right md:flex-row gap-x-10 gap-y-5 w-2/5 md:w-auto md:pt-2">
          {/* these routes are on the main mlflow.org site, which is hosted in a different repo */}
          <FooterMenuItem href="https://mllfow.org">Components</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/releases">Releases</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/blog">Blog</FooterMenuItem>
          <FooterMenuItem href={useBaseUrl("/")}>Docs</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org//ambassadors">
            Ambassador Program
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};
