import styles from "./simple-card.module.css";

export function SimpleGrid({ children }): JSX.Element {
  console.log("styles", styles);
  return <article className={styles.simpleGrid}>{children}</article>;
}

export function SimpleCard({ headerText, link, text }): JSX.Element {
  return (
    <div className={styles.simpleCard}>
      <a href={link}>
        <SimpleCardHeader>{headerText}</SimpleCardHeader>
        <p>{text}</p>
      </a>
    </div>
  );
}

function SimpleCardHeader({ children }): JSX.Element {
  return <div className={styles.header}>{children}</div>;
}
