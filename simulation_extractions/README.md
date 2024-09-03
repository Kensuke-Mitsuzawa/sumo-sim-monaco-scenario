# What's this?

This project is for extracting sensor-information from SUMO simulations where the study scenario is ["Monaco scenario"](https://github.com/lcodeca/MoSTScenario).

The "sensor-information" is an array of `(|S|, |T|)`, where `S` is a set of sensors, `|T|` is a set of timestamps. 
Note that I use `lane` and `edge` as a sensor.
The Monaco scenario does not have enough detectors.

<!-- "trajectory-information" is an array of `(|A|, |T|)`, where `A` is a set of agents (vehivle, public transportation etc.) -->

# What can do?

- Executing SUMO simulations.
- Making an aggregated file.
- Making visualizations.

# Script Guide


| Script Name                         | Usage Type      | Summary                                                   | Output                     | Directory                                 | status         |
|-------------------------------------|-----------------|-----------------------------------------------------------|----------------------------|-------------------------------------------|----------------|
| task_pipeline.py                    | simulation-exe. | executing SUMO simulation                                 | SUMO related-files         | .                                         | active         |
| make_heavy_blocking_scenario.py     | simulation-exe. | generating the SUMO simulation scenario files             | SUMO-related configs       | ./script_scenario_construction            | active         |
| generate_kepler_interactive_tool.py | analysis        | generating geo-csv files for "Foursquare" and "Kepler.gl" | geo-csv                    | ./simulation_extractions/cli_tools/export | active         |
| visualization_basic_statistics.py   | analysis        | visualising simple statistic                              | png files jsonl            | ./cli_tools/simple_analysis               | active         |
| generate_video.py                   | analysis        | generating a video file                                   | GIF and bunch of png files | ./cli_tools/simple_analysis               | ????           |
| make_aggregation.py                 | analysis        |                                                           | png files                  | ./cli_tools/simple_analysis               |                |
| visualize_network.py                | analysis        | visualising the study map                                 | png                        | ./cli_tools/export                        | not maintained |
| export_to_gis_tools.py              | analysis        | generating JSON files for an external GIS tool            | JSON                       | ./simulation_extractions/cli_tools/export | not maintained |


# Setup

`poetry install`


# How to use?


1. (Optional) Generating Scenario Files
2. Executing Simulations

## Generating Scenario Files

You generate the scenario configuration files if you need it.

## Executing Simulations

A toml file is necessary. 
`task_pipeline.py` is an interface.




