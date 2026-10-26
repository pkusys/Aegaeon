__all__ = ["syn_workload"]


def __getattr__(name):
    if name == "syn_workload":
        from aegaeon.workload import syn_workload

        return syn_workload
    raise AttributeError(f"module 'benchmark' has no attribute {name!r}")
