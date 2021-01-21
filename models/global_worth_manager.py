from models.worth_manager import WorthManager

_global_default_worth_manager = WorthManager()


def get_worth_manager():
    return _global_default_worth_manager


def reset_global_worth_manager():
    global _global_default_worth_manager
    _global_default_worth_manager = WorthManager()