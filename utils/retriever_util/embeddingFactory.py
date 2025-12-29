import importlib
import pkgutil
import os

class EmbeddingFactory:
    _class_map = {}
    _instance_cache = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._class_map[name] = subclass
            return subclass
        return decorator

    # @classmethod
    # def create(cls, name: str, *args, **kwargs):
    #     retriever_cls = cls._class_map.get(name)
    #     if retriever_cls is None:
    #         raise ValueError(f"Unknown retriever: {name}")
    #     return retriever_cls(*args, **kwargs)

    @classmethod
    def create(cls, name: str, singleton: bool = False, **kwargs):
        """
        创建实例
        
        Args:
            name: 注册的类名
            singleton: 是否使用单例模式，默认False（每次创建新实例）
            *args, **kwargs: 传递给类构造函数的参数
        """
        retriever_cls = cls._class_map.get(name)
        if retriever_cls is None:
            raise ValueError(f"Unknown retriever: {name}")
        
        # 如果启用单例模式
        if singleton:
            if name not in cls._instance_cache:
                cls._instance_cache[name] = retriever_cls(**kwargs)
            return cls._instance_cache[name]
        
        # 默认：每次创建新实例
        return retriever_cls(*args, **kwargs)
        
    @classmethod
    def auto_import_all(cls, package):
        """自动导入package目录下所有模块，确保所有子类都注册"""
        package_dir = os.path.dirname(package.__file__)
        for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
            if not is_pkg:
                importlib.import_module(f"{package.__name__}.{module_name}")

