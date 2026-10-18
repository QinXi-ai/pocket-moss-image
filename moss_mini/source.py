"""Pinned Hub source resolution and explicit local model identity."""
from pathlib import Path

MODEL = "OpenMOSS-Team/MOSS-Image-SFT"
REVISION = "797fa8e678b4f0a766271bf5103e6cd7a2faa999"


def resolve_source(model, revision, source_version, phase, offline=False):
    root = Path(model).expanduser()
    if root.is_dir():
        if not source_version:
            raise ValueError("Local models require --source-version; change it when weights/code/tokenizer change")
        identity = {"kind": "local", "version": source_version}
    else:
        if source_version:
            raise ValueError("--source-version is only for local models")
        from huggingface_hub import snapshot_download
        components = ["text_encoder", "tokenizer"] if phase == "encode" else ["transformer", "scheduler", "vae"]
        root = Path(snapshot_download(model, revision=revision, local_files_only=offline,
                    allow_patterns=["*.py", "model_index.json", *[f"{p}/*" for p in components]]))
        identity = {"kind": "hub", "repo_id": model, "revision": root.name}
    if not (root / "pipeline_moss_image.py").is_file():
        raise ValueError("Source must include the official pipeline_moss_image.py")
    return root.resolve(), identity


def pipeline_class(source):
    from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
    return get_class_from_dynamic_module(str(source), module_file="pipeline_moss_image.py",
        class_name="MOSSImagePipeline", local_files_only=True, trust_remote_code=True)
