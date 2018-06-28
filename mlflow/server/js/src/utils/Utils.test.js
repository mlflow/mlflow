import Utils from './Utils'

test("formatMetric", () => {
  expect(Utils.formatMetric(0)).toBe("0");
  expect(Utils.formatMetric(0.5)).toBe("0.5");
  expect(Utils.formatMetric(0.001)).toBe("0.001");

  expect(Utils.formatMetric(0.12345)).toBe("0.123");
  expect(Utils.formatMetric(0.12355)).toBe("0.124");
  expect(Utils.formatMetric(-0.12345)).toBe("-0.123");
  expect(Utils.formatMetric(-0.12355)).toBe("-0.124");

  expect(Utils.formatMetric(1.12345)).toBe("1.123");
  expect(Utils.formatMetric(1.12355)).toBe("1.124");
  expect(Utils.formatMetric(-1.12345)).toBe("-1.123");
  expect(Utils.formatMetric(-1.12355)).toBe("-1.124");

  expect(Utils.formatMetric(12.12345)).toBe("12.12");
  expect(Utils.formatMetric(12.12555)).toBe("12.13");
  expect(Utils.formatMetric(-12.12345)).toBe("-12.12");
  expect(Utils.formatMetric(-12.12555)).toBe("-12.13");

  expect(Utils.formatMetric(123.12345)).toBe("123.1");
  expect(Utils.formatMetric(123.15555)).toBe("123.2");
  expect(Utils.formatMetric(-123.12345)).toBe("-123.1");
  expect(Utils.formatMetric(-123.15555)).toBe("-123.2");

  expect(Utils.formatMetric(1234.12345)).toBe("1234.1");
  expect(Utils.formatMetric(1234.15555)).toBe("1234.2");
  expect(Utils.formatMetric(-1234.12345)).toBe("-1234.1");
  expect(Utils.formatMetric(-1234.15555)).toBe("-1234.2");

  expect(Utils.formatMetric(1e30)).toBe("1e+30");
});

test("formatDuration", () => {
  expect(Utils.formatDuration(0)).toBe("0ms");
  expect(Utils.formatDuration(50)).toBe("50ms");
  expect(Utils.formatDuration(499)).toBe("499ms");
  expect(Utils.formatDuration(500)).toBe("0.5s");
  expect(Utils.formatDuration(900)).toBe("0.9s");
  expect(Utils.formatDuration(999)).toBe("1.0s");
  expect(Utils.formatDuration(1000)).toBe("1.0s");
  expect(Utils.formatDuration(1500)).toBe("1.5s");
  expect(Utils.formatDuration(2000)).toBe("2.0s");
  expect(Utils.formatDuration(59 * 1000)).toBe("59.0s");
  expect(Utils.formatDuration(60 * 1000)).toBe("1.0min");
  expect(Utils.formatDuration(90 * 1000)).toBe("1.5min");
  expect(Utils.formatDuration(120 * 1000)).toBe("2.0min");
  expect(Utils.formatDuration(59 * 60 * 1000)).toBe("59.0min");
  expect(Utils.formatDuration(60 * 60 * 1000)).toBe("1.0h");
  expect(Utils.formatDuration(90 * 60 * 1000)).toBe("1.5h");
  expect(Utils.formatDuration(23 * 60 * 60 * 1000)).toBe("23.0h");
  expect(Utils.formatDuration(24 * 60 * 60 * 1000)).toBe("1.0d");
  expect(Utils.formatDuration(36 * 60 * 60 * 1000)).toBe("1.5d");
  expect(Utils.formatDuration(48 * 60 * 60 * 1000)).toBe("2.0d");
  expect(Utils.formatDuration(480 * 60 * 60 * 1000)).toBe("20.0d");
});

test("formatUser", () => {
  expect(Utils.formatUser("bob")).toBe("bob");
  expect(Utils.formatUser("bob.mcbob")).toBe("bob.mcbob");
  expect(Utils.formatUser("bob@example.com")).toBe("bob");
});

test("baseName", () => {
  expect(Utils.baseName("foo")).toBe("foo");
  expect(Utils.baseName("foo/bar/baz")).toBe("baz");
  expect(Utils.baseName("/foo/bar/baz")).toBe("baz");
  expect(Utils.baseName("file:///foo/bar/baz")).toBe("baz");
});

test("dropExtension", () => {
  expect(Utils.dropExtension("foo")).toBe("foo");
  expect(Utils.dropExtension("foo.xyz")).toBe("foo");
  expect(Utils.dropExtension("foo.xyz.zyx")).toBe("foo.xyz");
  expect(Utils.dropExtension("foo/bar/baz.xyz")).toBe("foo/bar/baz");
  expect(Utils.dropExtension(".foo/.bar/baz.xyz")).toBe(".foo/.bar/baz");
  expect(Utils.dropExtension(".foo")).toBe(".foo");
  expect(Utils.dropExtension(".foo.bar")).toBe(".foo");
  expect(Utils.dropExtension("/.foo")).toBe("/.foo");
  expect(Utils.dropExtension(".foo/.bar/.xyz")).toBe(".foo/.bar/.xyz");
});
