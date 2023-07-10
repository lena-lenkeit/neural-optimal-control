import abc
from functools import partial
from typing import List, Optional, Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree, Scalar

import optimal_control.controls as controls


class AbstractConstraint(eqx.Module):
    ...


class AbstractProjectionConstraint(AbstractConstraint):
    @abc.abstractmethod
    def project(self, values: PyTree, times: PyTree) -> PyTree:
        ...


class AbstractLocalTransformationConstraint(AbstractConstraint):
    @abc.abstractmethod
    def transform_single(self, values: PyTree) -> PyTree:
        ...

    def transform_series(self, values: PyTree, times: PyTree) -> PyTree:
        return jax.vmap(self.transform_single)(values)


class AbstractGlobalTransformationConstraint(AbstractConstraint):
    @abc.abstractmethod
    def transform_series(self, values: PyTree, times: PyTree) -> PyTree:
        ...


class AbstractPenaltyConstraint(AbstractConstraint):
    penalty_weight: Optional[Scalar] = None


class AbstractLocalPenaltyConstraint(AbstractPenaltyConstraint):
    @abc.abstractmethod
    def penalty_single(self, values: PyTree) -> PyTree:
        ...

    def penalty_series(self, values: PyTree) -> PyTree:
        return jax.vmap(self.penalty_single)(values)

    def penalty_ode_term(self, values: PyTree) -> PyTree:
        return self.penalty_single(values)

    def penalty_ode_reward(self, integrated_term: PyTree) -> Scalar:
        return integrated_term * self.penalty_weight


class AbstractGlobalPenaltyConstraint(AbstractPenaltyConstraint):
    @abc.abstractmethod
    def penalty_series(self, values: PyTree) -> PyTree:
        ...

    @abc.abstractmethod
    def penalty_ode_term(self, values: PyTree) -> PyTree:
        ...

    @abc.abstractmethod
    def penalty_ode_reward(self, integrated_term: PyTree) -> Scalar:
        ...


class ConstraintChain(eqx.Module):
    projections: List[AbstractProjectionConstraint]
    transformations: List[AbstractGlobalTransformationConstraint]
    penalties: List[AbstractPenaltyConstraint]
