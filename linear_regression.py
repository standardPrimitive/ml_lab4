from __future__ import annotations

from typing import List

import numpy as np
from numpy import linalg

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        #x = np.hstack([np.ones((x.shape[0], 1)), x])  # Добавляем столбец с единицами для свободного члена

        # prev_weights = self.descent.w.copy()
        # for iteration in range(self.max_iter):
        #     self.loss_history.append(self.calc_loss(x, y))
        #     weights_difference = self.descent.step(x, y)
        #
        #     # Критерий остановки: евклидова норма разности векторов весов
        #     if np.linalg.norm(weights_difference) < self.tolerance:
        #         break
        #     # Критерий остановки: NaN значения в весах
        #     if np.isnan(self.descent.w).any():
        #         print(f"Обнаружено NaN значение на итерации {iteration}")
        #         break
        #     # Проверка изменения весов
        #     if np.allclose(self.descent.w, prev_weights, atol=self.tolerance):
        #         print(f"Сходимость достигнута на итерации {iteration}")
        #         break
        #
        #     prev_weights = self.descent.w.copy()
        #
        # # Добавляем последнее значение функции потерь после выхода из цикла
        # self.loss_history.append(self.calc_loss(x, y))
        # return self

        self.loss_history += [self.descent.calc_loss(x, y)]
        zero_step_weights = self.descent.step(x, y)
        tolerance = (linalg.norm(zero_step_weights, ord=2)) ** 2
        is_diff_nan = np.isnan(zero_step_weights).sum()
        self.max_iter -= 1

        while (self.max_iter > 0 and is_diff_nan == 0 and self.tolerance < tolerance):
            self.loss_history += [self.descent.calc_loss(x, y)]
            old_w = self.descent.w
            zero_step_weights = self.descent.step(x, y)
            tolerance = (linalg.norm(zero_step_weights, ord=2)) ** 2
            is_diff_nan = np.isnan(zero_step_weights).sum()
            self.max_iter -= 1
        self.loss_history += [self.descent.calc_loss(x, y)]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)
