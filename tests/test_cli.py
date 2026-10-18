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


def test_packed_hub_resolution_excludes_original_encoder_shards(monkeypatch, tmp_path):
    import sys
    import types
    (tmp_path / "pipeline_moss_image.py").write_text("# test")
    calls = []
    def download(model, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=download))
    resolve_source('organization/model', 'revision', None, 'encode', encoder_weights=False)
    patterns = calls[-1]['allow_patterns']
    assert 'text_encoder/config.json' in patterns and 'tokenizer/*' in patterns
    assert 'text_encoder/*' not in patterns
    resolve_source('organization/model', 'revision', None, 'encode')
    assert 'text_encoder/*' in calls[-1]['allow_patterns']
    resolve_source('organization/model', 'revision', None, 'generate', generator_weights=False)
    patterns = calls[-1]['allow_patterns']
    assert 'transformer/config.json' in patterns and 'transformer/*' not in patterns
    assert 'vae/*' in patterns and 'scheduler/*' in patterns


def test_pack_generator_parser_and_paths():
    args = parser().parse_args(['pack-generator', '--output', 'outputs/packed-generator'])
    assert args.command == 'pack-generator'
    args = parser().parse_args(['generate', '--prompt', 'a cat', '--output', 'outputs/cat',
                               '--packed-generator', 'models/dit'])
    assert str(args.packed_generator).replace('\\', '/') == 'models/dit'


def test_acceleration_options_are_explicit():
    base = ['generate', '--prompt', 'cat', '--output', 'outputs/cat']
    args = parser().parse_args(base)
    assert not args.request_cache and not args.cache_dit and args.embedding_lookup == 'offload'
    args = parser().parse_args(base + ['--request-cache', '--cache-dit', '--cache-threshold', '0.1',
                                     '--embedding-lookup', 'cpu', '--offload-layers', '4'])
    assert args.request_cache and args.cache_dit and args.cache_threshold == .1
    assert args.embedding_lookup == 'cpu' and args.offload_layers == 4
