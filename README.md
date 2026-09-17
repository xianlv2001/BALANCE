# BALANCE
Boosting Index Advisor Learning with Multi-Source Workload Knowledge

![Framework overview of BALANCE](./image.png)

Above is the overall architecture of BALANCE, and the code runs on Postgresql 12.5.
### Code structure
```
BALANCE/
│
├── main.py                       # Entry point: builds the experiment and starts PPO training
├── requirements.txt              # Python dependencies
├── image.png                     # Architecture figure shown above
├── box_line.pickle               # Pre-computed predicate value boxes (used by src/plan_encoding)
├── tpcds_lsi.model               # Pre-trained LSI workload model (+ .projection file)
│
├── balance/                      # Core package: experiment setup and RL components
│   ├── experiment.py             #   Experiment: config, schema, workloads, envs, train/evaluate
│   ├── configuration_parser.py   #   Parses the JSON experiment configuration
│   ├── schema.py                 #   Database schema (tables/columns) on PostgreSQL
│   ├── workload_generator.py     #   Samples training/validation workloads
│   ├── workload_embedder.py      #   Encodes workloads as vectors (plan-based LSI-BOW, SQL)
│   ├── embedding_utils.py        #   Helpers for pruning queries during embedding
│   ├── boo.py                    #   Bag of Operators: query plans -> operator sets
│   ├── observation_manager.py    #   Builds RL observations (embedded plan + cost features)
│   ├── action_manager.py         #   Index actions: valid actions under the storage budget
│   ├── reward_calculator.py      #   Reward from cost difference vs. storage consumption
│   └── utils.py                  #   Shared utilities
│
├── gym_db/                       # Gym environment for index selection
│   ├── common.py                 #   EnvironmentType: training / validation / testing
│   └── envs/
│       └── db_env_v1.py          #   DBEnvV1: one episode = recommend indexes -> measure cost -> reward
│
├── src/                          # Value embedder (multi-source workload knowledge)
│   ├── parameters.py             #   Global dimensions and IDs of tables/columns/operators
│   ├── feature_extraction/       #   Feature extraction from plans, predicates, and bitmaps
│   ├── plan_encoding/            #   Query-plan tree encoding
│   └── token_embedding/          #   Word2Vec embedding of query tokens and values
│
├── stable_baselines/             # Bundled OpenAI Baselines (v2)
│   └── ppo2/
│       └── ppo2_BALANCE.py       #   PPO2 adapted for BALANCE (used by main.py)
│
├── experiments/                  # Experiment configurations
│   ├── tpch.json                 #   Settings for TPCH
│   └── tpcds.json                #   Settings for TPCDS
│
├── experiment_results/           # Saved models and generated workloads
│   ├── cl_save/gen_model/        #   Pre-trained CL model checkpoint (.pth)
│   ├── source/                   #   Pre-trained source models for transfer (main.py loads f_s1-f_s3.zip here)
│   └── workloads/gen_tpch/       #   Generated TPCH training workloads (.pickle)
│
├── query_files/                  # Raw benchmark query texts
│   ├── TPCH/                     #   TPCH_1.txt - TPCH_22.txt
│   └── TPCDS/                    #   TPCDS_1.txt - TPCDS_99.txt
│
└── index_selection_evaluation/   # Database toolkit: connectors, what-if cost evaluation, benchmark kits
```

Training flow: `main.py` -> `balance/experiment.py` (setup) -> `gym_db` environment -> `stable_baselines/ppo2/ppo2_BALANCE.py` (PPO2), with `src/` supplying the value embedder and `index_selection_evaluation/` the database utilities.

### Example workflow

```
pip install -r requirements.txt         # Install requirements with pip
python main.py                          # Run a experiment
```
Experiments can be controlled with the **./experiments/tpch.json** file. For descriptions of the components and functioning, consult our paper.
