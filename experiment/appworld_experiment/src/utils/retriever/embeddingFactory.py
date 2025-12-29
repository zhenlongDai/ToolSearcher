import importlib
import pkgutil
import os

class EmbeddingFactory:
    _class_map = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._class_map[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        retriever_cls = cls._class_map.get(name)
        if retriever_cls is None:
            raise ValueError(f"Unknown retriever: {name}")
        return retriever_cls(*args, **kwargs)

    @classmethod
    def auto_import_all(cls, package):
        """自动导入package目录下所有模块，确保所有子类都注册"""
        package_dir = os.path.dirname(package.__file__)
        for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
            if not is_pkg:
                importlib.import_module(f"{package.__name__}.{module_name}")

