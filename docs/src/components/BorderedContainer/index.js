import styles from './styles.module.css';
export default function BorderedContainer(_a) {
    var children = _a.children;
    return <section className={styles.Container}>{children}</section>;
}
