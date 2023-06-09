# Curious Agents
This repo explores methods of training RL agents using unsupervised and self-supervised leanring. The main idea is that an agent first explores and learns to understand an environment without explicit rewards provided to it. Then afterwards it can be fine-tuned using a reward signal. The goal is 1.) to see if this approach works in large open world examples and 2.) to see whether this approach scales better than training from scratch.

See [this blogpost](https://medium.com/@dries.epos/curious-agents-ebfee02ef024) for more information.

The base algorithm was adapted from https://github.com/luchris429/purejaxrl. Please check their GitHub repo out.
