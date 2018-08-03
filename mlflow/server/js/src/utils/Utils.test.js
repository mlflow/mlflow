import Utils from './Utils'

test("formatMetric", () => {
  expect(Utils.formatMetric(0)).toEqual("0");
  expect(Utils.formatMetric(0.5)).toEqual("0.5");
  expect(Utils.formatMetric(0.001)).toEqual("0.001");

  expect(Utils.formatMetric(0.12345)).toEqual("0.123");
  expect(Utils.formatMetric(0.12355)).toEqual("0.124");
  expect(Utils.formatMetric(-0.12345)).toEqual("-0.123");
  expect(Utils.formatMetric(-0.12355)).toEqual("-0.124");

  expect(Utils.formatMetric(1.12345)).toEqual("1.123");
  expect(Utils.formatMetric(1.12355)).toEqual("1.124");
  expect(Utils.formatMetric(-1.12345)).toEqual("-1.123");
  expect(Utils.formatMetric(-1.12355)).toEqual("-1.124");

  expect(Utils.formatMetric(12.12345)).toEqual("12.12");
  expect(Utils.formatMetric(12.12555)).toEqual("12.13");
  expect(Utils.formatMetric(-12.12345)).toEqual("-12.12");
  expect(Utils.formatMetric(-12.12555)).toEqual("-12.13");

  expect(Utils.formatMetric(123.12345)).toEqual("123.1");
  expect(Utils.formatMetric(123.15555)).toEqual("123.2");
  expect(Utils.formatMetric(-123.12345)).toEqual("-123.1");
  expect(Utils.formatMetric(-123.15555)).toEqual("-123.2");

  expect(Utils.formatMetric(1234.12345)).toEqual("1234.1");
  expect(Utils.formatMetric(1234.15555)).toEqual("1234.2");
  expect(Utils.formatMetric(-1234.12345)).toEqual("-1234.1");
  expect(Utils.formatMetric(-1234.15555)).toEqual("-1234.2");

  expect(Utils.formatMetric(1e30)).toEqual("1e+30");
});

test("formatDuration", () => {
  expect(Utils.formatDuration(0)).toEqual("0ms");
  expect(Utils.formatDuration(50)).toEqual("50ms");
  expect(Utils.formatDuration(499)).toEqual("499ms");
  expect(Utils.formatDuration(500)).toEqual("0.5s");
  expect(Utils.formatDuration(900)).toEqual("0.9s");
  expect(Utils.formatDuration(999)).toEqual("1.0s");
  expect(Utils.formatDuration(1000)).toEqual("1.0s");
  expect(Utils.formatDuration(1500)).toEqual("1.5s");
  expect(Utils.formatDuration(2000)).toEqual("2.0s");
  expect(Utils.formatDuration(59 * 1000)).toEqual("59.0s");
  expect(Utils.formatDuration(60 * 1000)).toEqual("1.0min");
  expect(Utils.formatDuration(90 * 1000)).toEqual("1.5min");
  expect(Utils.formatDuration(120 * 1000)).toEqual("2.0min");
  expect(Utils.formatDuration(59 * 60 * 1000)).toEqual("59.0min");
  expect(Utils.formatDuration(60 * 60 * 1000)).toEqual("1.0h");
  expect(Utils.formatDuration(90 * 60 * 1000)).toEqual("1.5h");
  expect(Utils.formatDuration(23 * 60 * 60 * 1000)).toEqual("23.0h");
  expect(Utils.formatDuration(24 * 60 * 60 * 1000)).toEqual("1.0d");
  expect(Utils.formatDuration(36 * 60 * 60 * 1000)).toEqual("1.5d");
  expect(Utils.formatDuration(48 * 60 * 60 * 1000)).toEqual("2.0d");
  expect(Utils.formatDuration(480 * 60 * 60 * 1000)).toEqual("20.0d");
});

test("formatUser", () => {
  expect(Utils.formatUser("bob")).toEqual("bob");
  expect(Utils.formatUser("bob.mcbob")).toEqual("bob.mcbob");
  expect(Utils.formatUser("bob@example.com")).toEqual("bob");
});

test("baseName", () => {
  expect(Utils.baseName("foo")).toEqual("foo");
  expect(Utils.baseName("foo/bar/baz")).toEqual("baz");
  expect(Utils.baseName("/foo/bar/baz")).toEqual("baz");
  expect(Utils.baseName("file:///foo/bar/baz")).toEqual("baz");
});

test("dropExtension", () => {
  expect(Utils.dropExtension("foo")).toEqual("foo");
  expect(Utils.dropExtension("foo.xyz")).toEqual("foo");
  expect(Utils.dropExtension("foo.xyz.zyx")).toEqual("foo.xyz");
  expect(Utils.dropExtension("foo/bar/baz.xyz")).toEqual("foo/bar/baz");
  expect(Utils.dropExtension(".foo/.bar/baz.xyz")).toEqual(".foo/.bar/baz");
  expect(Utils.dropExtension(".foo")).toEqual(".foo");
  expect(Utils.dropExtension(".foo.bar")).toEqual(".foo");
  expect(Utils.dropExtension("/.foo")).toEqual("/.foo");
  expect(Utils.dropExtension(".foo/.bar/.xyz")).toEqual(".foo/.bar/.xyz");
});

test("gitHubRegexMatch", () => {
  const gitHubRegex = Utils.getGitHubRegex();
  const urlAndExpected = [
    ["http://github.com/mlflow/mlflow-apps", ["/github.com/mlflow/mlflow-apps", "mlflow", "mlflow-apps"]],
    ["https://github.com/mlflow/mlflow-apps", ["/github.com/mlflow/mlflow-apps", "mlflow", "mlflow-apps"]],
    ["http://github.com/mlflow/mlflow-apps.git", ["/github.com/mlflow/mlflow-apps", "mlflow", "mlflow-apps"]],
    ["https://github.com/mlflow/mlflow-apps.git", ["/github.com/mlflow/mlflow-apps", "mlflow", "mlflow-apps"]],
    ["git@github.com:mlflow/mlflow-apps.git", ["/github.com/mlflow/mlflow-apps", "mlflow", "mlflow-apps"]],
    ["https://some-other-site.com?q=github.com/mlflow/mlflow-apps.git", null],
    ["ssh@some-server:mlflow/mlflow-apps.git", null],
  ]
  urlAndExpected.forEach(lst => {
    const url = lst[0];
    const match = url.match(gitHubRegex);
    expect(match == lst[1]);
  })
})
