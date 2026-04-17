-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.ab_test_results (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  run_id text,
  prompt text,
  use_case text,
  group_label text,
  model_used text,
  recommended_model text,
  eval_llama4_score double precision,
  eval_mistral_score double precision,
  eval_nova_score double precision,
  avg_accuracy_score double precision,
  score_stdev double precision,
  latency_ms integer,
  cost double precision,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT ab_test_results_pkey PRIMARY KEY (id)
);
CREATE TABLE public.benchmark_results (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  created_at timestamp with time zone DEFAULT now(),
  provider text NOT NULL,
  model_id text NOT NULL,
  use_case text,
  prompt text NOT NULL,
  prompt_complexity text CHECK (prompt_complexity = ANY (ARRAY['low'::text, 'mid'::text, 'high'::text])),
  response text,
  accuracy_score integer CHECK (accuracy_score >= 0 AND accuracy_score <= 100),
  cost double precision,
  tokens integer,
  latency_ms integer,
  clarity text,
  avg_accuracy_score double precision,
  eval_llama4_maverick_score integer,
  eval_mistral_large_score integer,
  eval_nova_premier_score integer,
  eval_deepseek_r1_score integer,
  score_stdev double precision,
  eval_conflict_flag boolean DEFAULT false,
  high_conflict_flag boolean DEFAULT false,
  low_confidence boolean DEFAULT false,
  confidence_level double precision,
  eval_count integer,
  prompt_hash text,
  invalid boolean DEFAULT false,
  syntax_pass boolean,
  syntax_checked boolean DEFAULT false,
  consistency_score double precision,
  win_rate double precision,
  domain text,
  has_ref_answer boolean DEFAULT false,
  reference_answer text,
  is_correct boolean,
  prompt_version integer DEFAULT 1,
  CONSTRAINT benchmark_results_pkey PRIMARY KEY (id)
);
CREATE TABLE public.model_priors (
  model_id text NOT NULL,
  use_case text NOT NULL,
  prompt_complexity text NOT NULL,
  prior_accuracy double precision,
  prior_cost double precision,
  prior_latency double precision,
  support_n integer,
  last_updated timestamp with time zone DEFAULT now(),
  CONSTRAINT model_priors_pkey PRIMARY KEY (model_id, use_case, prompt_complexity)
);
CREATE TABLE public.model_win_rates (
  model_id text NOT NULL,
  use_case text NOT NULL,
  complexity text NOT NULL,
  win_rate double precision,
  total_matches integer NOT NULL,
  wins integer NOT NULL,
  losses integer NOT NULL,
  ties integer NOT NULL,
  judge_count integer,
  last_updated timestamp with time zone DEFAULT now(),
  total_participations integer,
  decisive_matches integer,
  tie_rate double precision,
  confidence double precision,
  CONSTRAINT model_win_rates_pkey PRIMARY KEY (model_id, use_case, complexity)
);
CREATE TABLE public.pairwise_results (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  prompt_hash text NOT NULL,
  use_case text NOT NULL,
  complexity text,
  model_a text NOT NULL,
  model_b text NOT NULL,
  response_a text,
  response_b text,
  winner text NOT NULL,
  winner_model text NOT NULL,
  loser_model text NOT NULL,
  judge_model text NOT NULL,
  reason text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT pairwise_results_pkey PRIMARY KEY (id)
);
CREATE TABLE public.prompt_embeddings (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  prompt_hash text NOT NULL UNIQUE,
  model_name text DEFAULT 'all-MiniLM-L6-v2'::text,
  created_at timestamp with time zone DEFAULT now(),
  embedding USER-DEFINED,
  CONSTRAINT prompt_embeddings_pkey PRIMARY KEY (id)
);
CREATE TABLE public.prompt_logs (
  id bigint NOT NULL DEFAULT nextval('prompt_logs_id_seq'::regclass),
  prompt text NOT NULL,
  use_case text NOT NULL,
  clarity text NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT prompt_logs_pkey PRIMARY KEY (id)
);
CREATE TABLE public.routing_log (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  request_id text,
  prompt_hash text,
  use_case text,
  complexity text,
  clarity text,
  recommended_model text,
  data_source text,
  knn_neighbors integer,
  filter_level text,
  expected_accuracy double precision,
  confidence double precision,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT routing_log_pkey PRIMARY KEY (id)
);