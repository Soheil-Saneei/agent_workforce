I would start with a homogeneous horizontal team of \(N=3\), then vary agent count and cognitive diversity independently. Three is large enough for specialization, disagreement, and majority dynamics, but small enough to inspect every interaction. MultiAgentBench also found its clearest gain moving from one to three agents, with diminishing returns afterward [[multiagentbench]].

## Core experimental design

Use a factorial design with four independent variables:

| Variable | Initial levels |
|---|---|
| Team size | \(N \in \{1,3,5\}\) |
| Approach diversity | identical, moderate, strong |
| Communication | none, horizontal shared discussion |
| Compute budget | fixed total team tokens |

Holding total tokens fixed is essential. Otherwise, “more agents” simply means “more inference compute.”

The clean initial experiment is:

\[
3\text{ team sizes}
\times
3\text{ diversity levels}
\times
2\text{ communication conditions}
\]

This lets you estimate:

1. The benefit of additional independent samples.
2. The benefit of deliberately different approaches.
3. The marginal benefit of communication.
4. Whether communication helps diverse agents more than identical agents.

## How different should the agents be?

Start by manipulating only the problem-solving procedure, not personality, model, information, or organizational authority.

### 1. Identical approach

All agents receive the same role and problem-solving instructions, with independent sampling seeds.

This measures the value of simple replication or “more draws from the same reasoner.”

### 2. Moderate diversity

Give agents distinct but broadly applicable reasoning procedures:

- Decomposition agent: break the problem into subproblems and compose the answer.
- Evidence-first agent: extract relevant facts before drawing conclusions.
- Falsification agent: generate candidates and actively search for counterexamples.

This should be the primary treatment. The approaches differ enough to produce nonredundant errors but remain capable of solving the entire task.

### 3. Strong diversity

Use narrower occupational roles:

- Solver
- Critic
- Alternative-hypothesis generator
- Fact checker
- Integrator

This tests specialization, but it introduces a confound: agents no longer perform interchangeable work. If strong diversity wins, it may be because of workflow complementarity rather than cognitive diversity.

My hypothesis is an inverted-U relationship:

\[
\text{team value}
\approx
\text{nonredundancy}
-
\text{integration cost}
\]

Identical agents produce correlated errors. Extremely specialized agents may fail to integrate. Moderate procedural diversity should perform best.

## Horizontal-team protocol

For the first experiment, avoid assigning a leader or final-answer authority. A clean protocol would be:

1. Every agent independently produces an initial solution.
2. All initial solutions are revealed simultaneously.
3. Each agent submits one critique or revision.
4. The team selects an answer through anonymous majority vote.
5. Ties use confidence-weighted voting with confidence elicited before discussion.

Simultaneous revelation prevents anchoring on the first speaker. Anonymous voting prevents an accidental hierarchy. Limiting discussion prevents the runaway token use seen in flat OrgAgent teams [[orgagent]].

Include three aggregation controls:

- Best individual oracle: did any agent find the correct answer?
- Majority vote: can the team select the answer?
- Independent ensemble: vote without communication.

These distinguish generation from coordination:

\[
\text{Generation gain}
=
\text{oracle team score}
-
\text{single-agent score}
\]

\[
\text{Selection gain}
=
\text{team score}
-
\text{average individual score}
\]

\[
\text{Coordination gain}
=
\text{discussion score}
-
\text{independent-vote score}
\]

The oracle gap is especially informative. If one agent often finds the answer but the team rejects it, the problem is social aggregation, not reasoning capacity.

## Benchmarks

Use at least three task families because organizational effects are task dependent. OrgAgent’s hierarchy helped substantially on MuSiQue and SQuAD 2.0 but lost to flat teams in two MuSR settings [[orgagent]].

### Recommended first suite

1. **MuSiQue**

   Multi-hop evidence composition. Good for measuring division of reasoning and evidence integration. It was responsive to organizational structure in OrgAgent.

2. **MuSR**

   Long-context soft reasoning. Useful because it produced different organizational conclusions from MuSiQue, protecting you from benchmark-specific claims.

3. **Executable coding tasks**

   Use HumanEval, MBPP, or a similarly contained test-based benchmark. Objective execution lets you measure whether discussion produces working artifacts rather than persuasive prose.

A strong fourth option is database diagnosis from MultiAgentBench. It naturally supports competing causal hypotheses and has an objective root-cause label [[multiagentbench]].

I would not begin with research ideation, bargaining, or Werewolf. Their evaluations involve subjective judges, strategic asymmetry, or unclear ground truth. They become valuable later when studying organizational behavior rather than establishing the basic causal effect.

## Measurements

Primary outcomes:

- Objective task accuracy or executable success
- Total input and output tokens
- Wall-clock latency
- Accuracy per 10,000 tokens

Team-process outcomes:

- Pairwise disagreement before discussion
- Pairwise answer convergence after discussion
- Error correlation between agents
- Unique correct solutions contributed by each approach
- Rate at which correct minority answers are overturned
- Communication redundancy
- Contribution inequality across agents

The diversity manipulation should be validated behaviorally. Do not assume different prompts created genuinely different approaches. Measure:

- overlap in cited evidence
- overlap in intermediate steps
- answer correlation
- semantic similarity between solutions
- counterfactual consistency of each agent’s assigned procedure

If two role prompts produce nearly identical reasoning traces and errors, they are labels, not real organizational differentiation.

## First experiment I would actually run

Use one strong base model throughout:

- \(N = 1, 3, 5\)
- identical versus moderately diverse agents
- independent vote versus two-round horizontal discussion
- MuSiQue, MuSR, and executable coding
- fixed total token budget per problem
- at least three stochastic replications per item
- temperature and context access held constant

This is small enough to interpret while identifying whether the important effect comes from team size, cognitive diversity, or communication. Only after that should you introduce workplace structures such as functional departments, matrix organization, hierarchy, rotating leadership, decentralization, or incentive conflict.
