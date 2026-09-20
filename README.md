# BALANCE
Boosting Index Advisor Learning with Multi-Source Workload Knowledge

![Framework overview of BALANCE](./image.png)

Above is the overall architecture of BALANCE, and the code runs on Postgresql 12.5.
### Code structure

Top level, grouped by role; the packages that carry most of the logic are expanded below.

```
BALANCE/
│
│  ── Code ─────────────────────────────
├── main.py                       # Entry point: builds the Experiment and starts PPO training
├── requirements.txt              # Python dependencies
├── balance/                      # Core package: experiment setup + RL components
├── gym_db/                       # Gym environment for index selection
├── src/                          # Value embedder: multi-source workload knowledge
├── stable_baselines/             # Bundled OpenAI Baselines (v2), adapted PPO2
├── index_selection_evaluation/   # DB toolkit: what-if cost evaluation, benchmark kits
│
│  ── Configs, data & assets ───────
├── experiments/                  # JSON experiment configs: tpch.json, tpcds.json
├── experiment_results/           # Saved models and generated workloads
├── query_files/                  # Raw benchmark queries: TPCH_1-22, TPCDS_1-99
├── image.png                     # Architecture figure shown above
├── box_line.pickle               # Pre-computed predicate value boxes (used by src/)
└── tpcds_lsi.model               # Pre-trained LSI workload model (+ .projection file)
```

**`balance/`** — experiment setup and RL components, grouped by responsibility:

```
balance/
│
│  ── Experiment setup ─────────────
├── experiment.py                 # Central Experiment class: config, schema, workloads, envs, train/evaluate
├── configuration_parser.py       # Parses the JSON experiment configuration
├── schema.py                     # Database schema (tables/columns) on PostgreSQL
│
│  ── Workload encoding ────────────
├── workload_generator.py         # Samples training/validation workloads
├── workload_embedder.py          # Encodes workloads as vectors (plan-based LSI-BOW, SQL)
├── embedding_utils.py            # Helpers for pruning queries during embedding
├── boo.py                        # Bag of Operators: query plans -> operator sets
│
│  ── RL interface ─────────────────
├── observation_manager.py        # Builds RL observations (embedded plan + cost features)
├── action_manager.py             # Valid index actions under the storage budget
├── reward_calculator.py          # Reward from cost difference vs. storage consumption
│
└── utils.py                      # Shared utilities
```

**`src/`, `gym_db/`, `stable_baselines/`** — value embedder, Gym environment, and the PPO2 learner:

```
src/
├── parameters.py                 # Global dimensions and IDs of tables/columns/operators
├── feature_extraction/           # Features from plans, predicates, and bitmaps
├── plan_encoding/                # Query-plan tree encoding
└── token_embedding/              # Word2Vec embedding of query tokens and values

gym_db/
├── common.py                     # EnvironmentType: training / validation / testing
└── envs/
    └── db_env_v1.py              # One episode = recommend indexes -> measure cost -> reward

stable_baselines/
└── ppo2/
    └── ppo2_BALANCE.py           # PPO2 adapted for BALANCE (used by main.py)
```

**`experiment_results/`, `query_files/`** — checkpoints, generated workloads and benchmark queries:

```
experiment_results/
├── cl_save/gen_model/            # Pre-trained CL model checkpoint (.pth)
├── source/                       # Pre-trained source models for transfer (main.py loads f_s1-f_s3.zip here)
└── workloads/gen_tpch/           # Generated TPCH training workloads (.pickle)

query_files/
├── TPCH/                         # TPCH_1.txt - TPCH_22.txt
└── TPCDS/                        # TPCDS_1.txt - TPCDS_99.txt
```

Training flow: `main.py` -> `balance/experiment.py` (setup) -> `gym_db` environment -> `stable_baselines/ppo2/ppo2_BALANCE.py` (PPO2), with `src/` supplying the value embedder and `index_selection_evaluation/` the database utilities.

### Example workflow

```
pip install -r requirements.txt         # Install requirements with pip
python main.py                          # Run a experiment
```
Experiments can be controlled with the **./experiments/tpch.json** file. For descriptions of the components and functioning, consult our paper.
