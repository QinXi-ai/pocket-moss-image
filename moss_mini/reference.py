"""Reference constants and Orbit workspace release hook from MOSS engineering."""
REVISION = "20261017-fused-int8-v1"
ORBIT_COMMIT = "895789d34d95682590ef1ee76ac43b62ef8c77a9"
RECIPE = "moss-orbit-fused-rowabsmax-int8-v1"


def _release_apply(self, fn, recurse=True):
    from orbitquant.fused import release
    release(self)
    runtime = self.__dict__.get("oq_runtime")
    if runtime is not None:
        runtime.__dict__.pop("moss_image_rope", None)
    return self.__dict__["_moss_opt_original_apply"](fn, recurse=recurse)
