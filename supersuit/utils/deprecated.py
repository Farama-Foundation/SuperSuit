class DeprecatedWrapper(ImportError):
    pass


class Deprecated:
    def __init__(self, wrapper_name, orig_version, new_version):
        self.name = wrapper_name
        self.old_version, self.new_version = orig_version, new_version

    def __call__(self, env, *args, **kwargs):
        raise DeprecatedWrapper(f"{self.name}_{self.old_version} is now Deprecated, use {self.name}_{self.new_version} instead")
