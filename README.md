# BALANCE

**BALANCE: Boosting Index Advisor Learning with Multi-Source Workload Knowledge**

BALANCE is a deep reinforcement learning (DRL) based **index advisor**. It treats index selection as a sequential decision process: an agent repeatedly proposes indexes for the workload, observes the cost change estimated by PostgreSQL, and learns to configure indexes that minimize workload cost under a given storage budget.

![Framework overview of BALANCE](./image.png)

- **Tested environment**: PostgreSQL 12.5 with the [HypoPG](https://hypopg.readthedocs.io/) extension (for what-if index cost estimation)
- **Benchmarks**: TPC-H and TPC-DS
- **RL toolkit**: OpenAI Baselines PPO2 (bundled in `stable_baselines/`)
- **Pre-trained models**: `main.py` loads three source models from `experiment_results/source/` (`f_s1.zip` - `f_s3.zip`); they are not included in the repo, so place your checkpoints there before running

## How it works

The training loop connects the modules below:

```
main.py                      parses experiments/tpch.json, loads pre-trained source models,
  |                          builds the PPO2 agent (stable_baselines/ppo2/ppo2_BALANCE.py)
  v
balance/experiment.py        sets up the schema, generates workloads, and creates the
  |                          training / validation / testing environments
  v
gym_db/envs/db_env_v1.py     one episode = one workload: the agent adds indexes step by
  |                          step; after each step PostgreSQL (via HypoPG) re-estimates
  |                          the workload cost and the reward is computed
  v
balance/                     closes the RL loop: observation_manager.py builds the
                             observation, action_manager.py masks invalid index actions,
                             reward_calculator.py computes the reward
```

`src/` provides the value embedder that turns query plans, predicates, and column values into dense features for the observation manager; `index_selection_evaluation/` (an included copy of the [Index Selection Evaluation](https://github.com/hyrise/index_selection_evaluation) toolkit) supplies the PostgreSQL connector, HypoPG what-if index creation, and comparison algorithms (e.g. Extend) underneath.

## Code structure

Grouped by role rather than alphabetically — top-down is also the training flow:
entry point → RL core → workload knowledge → database toolkit → data & assets.

```
BALANCE/
│
│  ── Entry & configuration ─────────────────────────────────
├── main.py                      # Entry point: builds the experiment and starts PPO training
├── experiments/                 # Experiment configs: tpch.json (TPC-H), tpcds.json (TPC-DS)
├── requirements.txt             # Python dependencies
│
│  ── RL core: environment, actions, observations, reward ────
├── balance/                     # Core package: experiment setup and RL components
│   │
│   │  ── experiment setup ──
│   ├── experiment.py            #   Central Experiment class: config, schema, workloads, envs
│   ├── configuration_parser.py  #   Parses and validates the JSON experiment configuration
│   ├── schema.py                #   Database schema (tables/columns) on PostgreSQL
│   │
│   │  ── workload encoding ──
│   ├── workload_generator.py    #   Samples training/validation workloads from query_files/
│   ├── workload_embedder.py     #   Encodes workloads as vectors (plan-based LSI-BOW, SQL)
│   ├── embedding_utils.py       #   Helpers for pruning queries during embedding
│   ├── boo.py                   #   Bag of Operators: query plans -> operator sets
│   │
│   │  ── RL interface ──
│   ├── observation_manager.py   #   Builds RL observations (embedded plan + cost features)
│   ├── action_manager.py        #   Valid index actions under the storage budget
│   ├── reward_calculator.py     #   Reward from cost difference vs. storage consumption
│   └── utils.py                 #   Shared utilities
│
├── gym_db/                      # Gym environment for index selection
│   ├── common.py                #   EnvironmentType: training / validation / testing
│   └── envs/db_env_v1.py        #   DBEnvV1: one episode = one workload, one step = one index
│
├── stable_baselines/            # Bundled OpenAI Baselines (v2) — only ppo2/ is used
│   └── ppo2/ppo2_BALANCE.py     #   PPO2 adapted for BALANCE (used by main.py)
│
│  ── Workload knowledge: value embedder ────────────────────
├── src/                         # Turns plans, predicates and column values into features
│   ├── parameters.py            #   Global dimensions and IDs of tables/columns/operators
│   ├── feature_extraction/      #   Feature extraction from plans, predicates, and bitmaps
│   ├── plan_encoding/           #   Query-plan tree encoding
│   └── token_embedding/         #   Word2Vec embedding of query tokens and values
│
│  ── Database toolkit ──────────────────────────────────────
├── index_selection_evaluation/  # DB toolkit: PostgreSQL connector, HypoPG what-if index
│                                #   creation, and comparison algorithms (e.g. Extend)
│
│  ── Data & assets ─────────────────────────────────────────
├── experiment_results/          # Saved models and generated workloads
│   ├── cl_save/gen_model/       #   Pre-trained CL model checkpoint (.pth)
│   ├── source/                  #   Source models for transfer (place f_s1-f_s3.zip here)
│   └── workloads/gen_tpch/      #   Generated TPCH training workloads (.pickle)
│
├── query_files/                 # Raw benchmark query texts
│   ├── TPCH/                    #   TPCH_1.txt - TPCH_22.txt
│   └── TPCDS/                   #   TPCDS_1.txt - TPCDS_99.txt
│
├── image.png                    # Architecture figure shown above
├── box_line.pickle              # Pre-computed predicate value boxes (used by src/plan_encoding)
└── tpcds_lsi.model              # Pre-trained LSI workload model (+ .projection file)
```

## Getting started

### 1. Prerequisites

- PostgreSQL 12.5 running locally, with the [HypoPG](https://hypopg.readthedocs.io/) extension installed and available (`create extension hypopg`)
- The connection settings (user / password / host / port) are hard-coded in `index_selection_evaluation/selection/dbms/postgres_dbms.py` - adjust them to your instance

### 2. Install

```
pip install -r requirements.txt         # Install requirements with pip
```

### 3. Run an experiment

```
python main.py                          # Run an experiment
```

Experiments are controlled with the **./experiments/tpch.json** file. The main options include:

| Option | Meaning |
| --- | --- |
| `database` | Name of the database to create/use for the benchmark |
| `workload.benchmark` | `TPCH` or `TPCDS`; queries are read from `query_files/` |
| `workload.path` | Workload file used for training (see `experiment_results/workloads/`) |
| `budgets.validation_and_testing` | Storage budgets (MB) under which the advisor must stay |
| `max_index_width` | Maximum number of columns per index |
| `workload_embedder.type` | Workload representation, e.g. `PlanEmbedderLSIBOW` |
| `timesteps` | Total PPO training steps |
| `validation_frequency` | Steps between validations (test/validation callbacks) |

During training, validation/testing results and model checkpoints (best / moving-average / final) are written to `<result_path>/ID_<experiment id>/` - `experiment_results/ID_Test_Experiment_1/` for the shipped config. TensorBoard logs go to `tensor_log/`:

```
tensorboard --logdir tensor_log         # Monitor training curves
```

For descriptions of the components and functioning, consult our paper.
