import styles from './styles.module.css';

export default function BorderedContainer({ children }): JSX.Element {
  return <section className={styles.Container}>{children}</section>;
}
