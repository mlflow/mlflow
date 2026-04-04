from dev.check_action_pins import ActionRef, _check_version_consistency


def _make_ref(action: str, sha: str, comment: str, location: str = "file.yml:1") -> ActionRef:
    return ActionRef(
        prefix=f"{location}: 'uses: {action}@{sha} # {comment}'",
        action=action,
        ref=sha,
        comment=comment,
    )


SHA1 = "a" * 40
SHA2 = "b" * 40


def test_no_errors_when_all_versions_consistent():
    refs = [
        _make_ref("actions/checkout", SHA1, "v4.1.0", ".github/workflows/lint.yml:10"),
        _make_ref("actions/checkout", SHA1, "v4.1.0", ".github/workflows/build.yml:5"),
    ]
    assert _check_version_consistency(refs) == []


def test_detects_mismatched_sha():
    refs = [
        _make_ref("actions/checkout", SHA1, "v4.1.0", ".github/workflows/lint.yml:10"),
        _make_ref("actions/checkout", SHA2, "v4.1.0", ".github/workflows/build.yml:5"),
    ]
    errors = _check_version_consistency(refs)
    assert len(errors) == 1
    assert "actions/checkout is pinned to multiple versions" in errors[0]
    assert ".github/workflows/lint.yml:10" in errors[0]
    assert ".github/workflows/build.yml:5" in errors[0]


def test_detects_mismatched_tag():
    refs = [
        _make_ref("actions/checkout", SHA1, "v4.1.0", ".github/workflows/lint.yml:10"),
        _make_ref("actions/checkout", SHA1, "v4.1.1", ".github/workflows/build.yml:5"),
    ]
    errors = _check_version_consistency(refs)
    assert len(errors) == 1
    assert "actions/checkout is pinned to multiple versions" in errors[0]


def test_independent_actions_not_flagged():
    refs = [
        _make_ref("actions/checkout", SHA1, "v4.1.0"),
        _make_ref("actions/setup-python", SHA2, "v5.0.0"),
    ]
    assert _check_version_consistency(refs) == []


def test_multiple_actions_with_inconsistencies():
    sha3 = "c" * 40
    refs = [
        _make_ref("actions/checkout", SHA1, "v4.1.0", ".github/workflows/a.yml:1"),
        _make_ref("actions/checkout", SHA2, "v4.1.1", ".github/workflows/b.yml:2"),
        _make_ref("actions/setup-python", sha3, "v5.0.0", ".github/workflows/a.yml:3"),
        _make_ref("actions/setup-python", SHA1, "v5.1.0", ".github/workflows/b.yml:4"),
    ]
    errors = _check_version_consistency(refs)
    assert len(errors) == 2
    action_names = [e.split(" is pinned")[0] for e in errors]
    assert "actions/checkout" in action_names
    assert "actions/setup-python" in action_names


def test_empty_refs():
    assert _check_version_consistency([]) == []


def test_single_ref_per_action():
    refs = [_make_ref("actions/checkout", SHA1, "v4.1.0")]
    assert _check_version_consistency(refs) == []
