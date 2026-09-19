
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import opfunu.cec_based.cec2017 as cec2017

    _HAS_OPFUNU = True
except ImportError:
    _HAS_OPFUNU = False

N_CEC2017_FUNCTIONS = 29


@dataclass
class BenchmarkFunction:
    name: str
    obj_func: Callable[[np.ndarray], float]
    dim: int
    lb: np.ndarray
    ub: np.ndarray
    f_global: float                                                                               


def using_cec2017() -> bool:
    return _HAS_OPFUNU


def _build_cec2017_function(index: int, dim: int) -> BenchmarkFunction:
    cls = getattr(cec2017, f"F{index}2017")
    f = cls(ndim=dim)
    return BenchmarkFunction(
        name=f"CEC2017_F{index}",
        obj_func=lambda x, _f=f: float(_f.evaluate(x)),
        dim=dim,
        lb=np.asarray(f.lb, dtype=float),
        ub=np.asarray(f.ub, dtype=float),
        f_global=float(f.f_global),
    )


def _build_classic_functions(dim: int) -> dict[str, BenchmarkFunction]:
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def rastrigin(x: np.ndarray) -> float:
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    def ackley(x: np.ndarray) -> float:
        d = len(x)
        return float(
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
            + 20
            + np.e
        )

    def griewank(x: np.ndarray) -> float:
        i = np.arange(1, len(x) + 1)
        return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)

    def rosenbrock(x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))

    def schwefel(x: np.ndarray) -> float:
        return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def zakharov(x: np.ndarray) -> float:
        i = np.arange(1, len(x) + 1)
        s = np.sum(0.5 * i * x)
        return float(np.sum(x**2) + s**2 + s**4)

    def michalewicz(x: np.ndarray, m: int = 10) -> float:
        i = np.arange(1, len(x) + 1)
        return float(-np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m)))

    def levy(x: np.ndarray) -> float:
        w = 1.0 + (x - 1.0) / 4.0
        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
        return float(term1 + term2 + term3)

    def dixon_price(x: np.ndarray) -> float:
        i = np.arange(2, len(x) + 1)
        return float((x[0] - 1.0) ** 2 + np.sum(i * (2.0 * x[1:] ** 2 - x[:-1]) ** 2))

                                    
    specs = [
        ("Sphere", sphere, -100.0, 100.0, 0.0),
        ("Rastrigin", rastrigin, -5.12, 5.12, 0.0),
        ("Ackley", ackley, -32.768, 32.768, 0.0),
        ("Griewank", griewank, -600.0, 600.0, 0.0),
        ("Rosenbrock", rosenbrock, -5.0, 10.0, 0.0),
        ("Schwefel", schwefel, -500.0, 500.0, 0.0),
        ("Zakharov", zakharov, -5.0, 10.0, 0.0),
        ("Michalewicz", michalewicz, 0.0, np.pi, None),                            
        ("Levy", levy, -10.0, 10.0, 0.0),
        ("DixonPrice", dixon_price, -10.0, 10.0, 0.0),
    ]
    return {
        name: BenchmarkFunction(
            name=name,
            obj_func=func,
            dim=dim,
            lb=np.full(dim, lb, dtype=float),
            ub=np.full(dim, ub, dtype=float),
            f_global=f_global,
        )
        for name, func, lb, ub, f_global in specs
    }


def get_function_names(dim: int = 30) -> list[str]:
    if _HAS_OPFUNU:
        return [f"CEC2017_F{i}" for i in range(1, N_CEC2017_FUNCTIONS + 1)]
    return list(_build_classic_functions(dim=dim).keys())


def build_function(name: str, dim: int = 30) -> BenchmarkFunction:
    if name.startswith("CEC2017_F"):
        if not _HAS_OPFUNU:
            raise RuntimeError("opfunu chưa được cài, không thể build hàm CEC2017.")
        index = int(name.replace("CEC2017_F", ""))
        return _build_cec2017_function(index, dim)

    classic = _build_classic_functions(dim=dim)
    if name not in classic:
        raise ValueError(f"Không tìm thấy benchmark function tên '{name}'.")
    return classic[name]
