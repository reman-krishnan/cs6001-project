"""dagame: game-theoretic analysis of deferred-authorization protocols (CS 6001 project).

Two independent implementations of the same game are provided so that each can audit the other:

* ``solver``  - fast recombining-lattice dynamic program (scales to thousands of blocks).
* ``tree``    - explicit extensive-form game (exponential size, small k only) with a generic
                backward-induction engine, exhaustive strategy enumeration, subgame-perfection
                verification and coalition (side-payment) analysis.

``checker`` builds the deviation checker on top of ``tree``. ``observations`` checks the closed-form results
against the solver and ``han_check`` compares the solver with the binomial option price.
"""
from .model import (Market, Trade, PayoffKind, ForfeitRule, Routing, Protocol, Scenario)
from .solver import solve, evaluate, Solution
from .tree import build_game
from .checker import check, assess_design

__all__ = ["Market", "Trade", "PayoffKind", "ForfeitRule", "Routing", "Protocol", "Scenario",
           "solve", "evaluate", "Solution", "build_game", "check", "assess_design"]
