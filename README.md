# Crafting Environment Benchmark

Benchmarking of RL and HRL agents on the [Crafting environment](https://github.com/IRLL/Crafting).

# Research objective

This work aims at evaluating the number of interaction steps and learning steps needed to succeed at hierarchical tasks provided by the Crafting environement (performance) in order to evaluate the following hypothesis:

1.  Hierarchical reinforcment learning (HRL) agents performs better that classic flat reinforcment learning agent (FRL) on some hierarchical tasks.
2.  Agents performance on a task is correlated to the either the learning or total complexity of given hierarchical behavior explanations (HBE) of the task as defined in the [HBE graphs](https://github.com/IRLL/options_graphs) framework.
3.  The difference of performance between HRL and FRL agents on a new task can be predicted using the difference of correlation between total and learning complexity for each agent on previous tasks.
