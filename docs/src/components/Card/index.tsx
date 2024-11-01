import clsx from "clsx";
import styles from "./card.module.css";

export const CardGroup = ({ children }): JSX.Element => (
  <div className={styles.CardGroup}>
    <div className={styles.CardGroupCards}>{children}</div>
  </div>
);

export const Card = ({ headerText, link, text }): JSX.Element => (
  <div className={clsx(styles.Card, styles.CardBordered)}>
    <a className={clsx(styles.Link)} title="headerText" href={link}>
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
    </a>
  </div>
);
