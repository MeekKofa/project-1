"""
Registry pattern for models and datasets.

Provides a centralized registration system for easy model/dataset discovery
and instantiation.
"""

from typing import Dict, Type, Any, Callable
import logging

logger = logging.getLogger(__name__)


class Registry:
    """Generic registry for storing and retrieving registered items."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, item: Any) -> Any:
        """
        Register an item with the given name.

        Args:
            name: Unique identifier for the item
            item: The item to register (class, function, etc.)

        Returns:
            The registered item (for use as decorator)
        """
        if name in self._registry:
            logger.warning(
                f"{self.name} '{name}' is already registered. Overwriting.")

        self._registry[name] = item
        logger.debug(f"Registered {self.name}: {name}")
        return item

    def get(self, name: str) -> Any:
        """
        Retrieve a registered item by name.

        Args:
            name: Name of the item to retrieve

        Returns:
            The registered item

        Raises:
            KeyError: If item not found
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"{self.name} '{name}' not found. "
                f"Available: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list:
        """List all registered item names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if an item is registered."""
        return name in self._registry

    def __repr__(self) -> str:
        items = ', '.join(self._registry.keys())
        return f"{self.name}Registry({items})"


# Global registries
MODEL_REGISTRY = Registry("Model")
DATASET_REGISTRY = Registry("Dataset")
LOSS_REGISTRY = Registry("Loss")


class ModelRegistry:
    """
    Model registry with factory methods.

    Usage:
        @ModelRegistry.register('yolov8')
        class YOLOv8Model(DetectionModelBase):
            ...

        # Later:
        model = ModelRegistry.build('yolov8', num_classes=2)
    """

    @staticmethod
    def register(name: str) -> Callable:
        """Decorator to register a model class."""
        def decorator(cls: Type) -> Type:
            MODEL_REGISTRY.register(name, cls)
            return cls
        return decorator

    @staticmethod
    def build(name: str, **kwargs) -> Any:
        """
        Build a model instance.

        Args:
            name: Name of the registered model
            **kwargs: Arguments to pass to model constructor

        Returns:
            Instantiated model
        """
        model_cls = MODEL_REGISTRY.get(name)
        return model_cls(**kwargs)

    @staticmethod
    def list_available() -> list:
        """List all registered models."""
        return MODEL_REGISTRY.list_available()

    @staticmethod
    def get_model_class(name: str) -> Type:
        """Get the model class without instantiating."""
        return MODEL_REGISTRY.get(name)


class DatasetRegistry:
    """
    Dataset registry with factory methods.

    Usage:
        @DatasetRegistry.register('detection')
        class DetectionDataset(Dataset):
            ...

        # Later:
        dataset = DatasetRegistry.build('detection', ...)
    """

    @staticmethod
    def register(name: str) -> Callable:
        """Decorator to register a dataset class."""
        def decorator(cls: Type) -> Type:
            DATASET_REGISTRY.register(name, cls)
            return cls
        return decorator

    @staticmethod
    def build(name: str, **kwargs) -> Any:
        """
        Build a dataset instance.

        Args:
            name: Name of the registered dataset
            **kwargs: Arguments to pass to dataset constructor

        Returns:
            Instantiated dataset
        """
        dataset_cls = DATASET_REGISTRY.get(name)
        return dataset_cls(**kwargs)

    @staticmethod
    def list_available() -> list:
        """List all registered datasets."""
        return DATASET_REGISTRY.list_available()


class LossRegistry:
    """Registry for loss functions."""

    @staticmethod
    def register(name: str) -> Callable:
        """Decorator to register a loss function."""
        def decorator(cls: Type) -> Type:
            LOSS_REGISTRY.register(name, cls)
            return cls
        return decorator

    @staticmethod
    def build(name: str, **kwargs) -> Any:
        """Build a loss function instance."""
        loss_cls = LOSS_REGISTRY.get(name)
        return loss_cls(**kwargs)

    @staticmethod
    def list_available() -> list:
        """List all registered loss functions."""
        return LOSS_REGISTRY.list_available()
