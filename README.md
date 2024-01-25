# What's this?

A docker environment for executing [Sumo Monaco Scenario](https://github.com/lcodeca/MoSTScenario).

Some utilities are together.


# Specs

A docker image creates the following environment,

- Ubuntu 20.04
- Sumo 1.14
- Python 3.10

# Python Interfaces

This project has an interface that you can run multiple-simulations in parallel.
This feature is useful when you wanna compare multiple scenarios.
The parallel computation is thanks to Dask.

See README in the `simulation_extractions`.

