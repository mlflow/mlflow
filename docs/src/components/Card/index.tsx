import clsx from "clsx";
import styles from "./card.module.css";
import Link from "@docusaurus/Link";

export const CardGroup = ({ children, isSmall }): JSX.Element => (
  <div
    className={clsx(
      styles.CardGroup,
      isSmall ? styles.AutofillColumns : styles.MaxThreeColumns
    )}
  >
    {children}
  </div>
);

export const Card = ({ children, link = "" }): JSX.Element => {
  if (!link) {
    return (
      <div className={clsx(styles.Card, styles.CardBordered)}>{children}</div>
    );
  }

  return (
    <Link
      className={clsx(styles.Link, styles.Card, styles.CardBordered)}
      to={link}
    >
      {children}
    </Link>
  );
};

export const PageCard = ({ headerText, link, text }): JSX.Element => (
  <Card link={link}>
    <span>
      <div
        className={clsx(
          styles.CardTitle,
          styles.BoxRoot,
          styles.PaddingBottom4
        )}
        style={{ pointerEvents: "none" }}
      >
        <div
          className={clsx(
            styles.BoxRoot,
            styles.FlexFlex,
            styles.FlexAlignItemsCenter,
            styles.FlexDirectionRow,
            styles.FlexJustifyContentFlexStart,
            styles.FlexWrapNowrap
          )}
          style={{ marginLeft: "-4px", marginTop: "-4px" }}
        >
          <div
            className={clsx(
              styles.BoxRoot,
              styles.BoxHideIfEmpty,
              styles.MarginTop4,
              styles.MarginLeft4
            )}
            style={{ pointerEvents: "auto" }}
          >
            <span className="">{headerText}</span>
          </div>
        </div>
      </div>
      <span className={clsx(styles.TextColor, styles.CardBody)}>
        <p>{text}</p>
      </span>
    </span>
  </Card>
);

export const LogoCard = ({ description, children, link }): JSX.Element => (
  <Card link={link}>
    <div className={styles.LogoCardContent}>
      <div className={styles.LogoCardImage}>{children}</div>
      <p className={styles.TextColor}>{description}</p>
    </div>
  </Card>
);

export const SmallLogoCard = ({ children, link }) => (
  <Card link={link}>
    <div className={styles.SmallLogoCardContent}>
      <div
        className={clsx("max-height-img-container", styles.SmallLogoCardImage)}
      >
        {children}
      </div>
    </div>
  </Card>
);
