import pytest
from moss_mini.cli import parser, OFFLOAD, main
from moss_mini.source import resolve_source


def test_default_recipe():
    args = parser().parse_args(["generate", "--prompt", "a cat", "--output", "outputs/cat"])
    assert args.bits == 6 and OFFLOAD[args.bits] == 20
    assert args.max_sequence_length == 256 and args.cpu_threads == 4


def test_local_identity_required(tmp_path):
    (tmp_path / "pipeline_moss_image.py").write_text("# test")
    with pytest.raises(ValueError, match="source-version"):
        resolve_source(str(tmp_path), None, None, "encode")
    root, identity = resolve_source(str(tmp_path), None, "test-v1", "encode")
    assert root == tmp_path.resolve() and identity == {"kind": "local", "version": "test-v1"}


def test_reject_long_context_before_model_load(tmp_path):
    with pytest.raises(ValueError, match="tokens"):
        main(["generate", "--prompt", "a cat", "--max-sequence-length", "4096", "--output", str(tmp_path/"out")])
    assert not (tmp_path/"out").exists()


def test_internal_worker_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr("moss_mini.cli.worker", lambda phase, path: calls.append((phase, str(path))))
    main(["_worker", "encode", "plan.json"])
    assert calls == [("encode", "plan.json")]
