# Product requirements

Status: implementation baseline  
Product scope: new desktop PC components sold in Singapore  
Currency: SGD  
Last updated: 2026-07-22

## Product outcome

PC Build Recommender turns a structured budget, workload mix, retained hardware, physical
constraints, and preferences into three to five distinct complete builds. A returned build is
useful only when its selected listings are within budget and in stock, all known hard
compatibility rules pass, missing evidence is visible, and the component-level reasons can be
traced to a data, rule, or model version.

This is an applied search-and-recommendation product. Hybrid retrieval and learned ranking find
good candidates; versioned deterministic rules decide compatibility; CP-SAT assembles the final
build. An LLM is not an authority for compatibility, price, benchmark, or availability claims.

## Users and jobs

| User | Primary job |
| --- | --- |
| First-time builder | Obtain a complete, explainable build without learning every interface and clearance rule. |
| Gamer | Balance target resolution and frame rate against total cost, noise, and power. |
| Local-AI user | Prioritise usable GPU memory, inference performance, software support, memory, cooling, and power. |
| Developer | Prioritise compilation, single- and multicore performance, memory, and responsive storage. |
| Content creator | Balance rendering, encoding, GPU acceleration, memory, and storage capacity. |
| Upgrade user | Retain owned parts and optimise only the compatible remainder. |

## Authoritative request contract

Structured form values are authoritative. Natural-language input may populate the same schema,
but the user must be able to inspect and change the parsed values before generation.

Required input:

- total budget in SGD;
- one or more workload profiles whose weights sum to one;
- retained components, if any;
- minimum GPU memory, system memory, and storage requirements;
- case-size, Wi-Fi, stock, and brand constraints;
- noise, efficiency, upgradeability, and brand preferences; and
- requested alternative profiles, from one to five.

Hard requirements and soft preferences must never be conflated. A preference can affect rank;
a hard requirement removes a candidate or makes the optimisation infeasible.

## Functional requirements

<!-- TODO: sections below still to be written. -->
