import styles from "./styles.module.css";

export default function Container({ children }): JSX.Element {
  return <section className={styles.leftBox}>{children}</section>;
}
