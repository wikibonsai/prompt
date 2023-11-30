# (Digital Gardening) Prompts

[![A WikiBonsai Project](https://img.shields.io/badge/%F0%9F%8E%8B-A%20WikiBonsai%20Project-brightgreen)](https://github.com/wikibonsai/wikibonsai)

These prompts are for generating useful notes in [digital gardens](https://github.com/wikibonsai/wikibonsai#notable-workflows). They are separated into modular units such that they may be mix'n'matched for various purposes.

These prompts have been tested with [OpenAI's ChatGPT](https://platform.openai.com/).

🤖 Ask a robot help you tend your 🎋 [WikiBonsai](https://github.com/wikibonsai/wikibonsai) digital garden.

## Design

Good prompts use "[controlled vocabulary](https://en.wikipedia.org/wiki/Controlled_vocabulary)". This is because words can be ambiguous, have different connotations based on context, or have multiple meanings -- which in linguistics are called "[word senses](https://en.wikipedia.org/wiki/Word_sense)".

In machine learning, all of this is represented in a [mathematical vector space](https://x.com/wibomd/status/1730255242564862008), or "semantic space". So, when prompting an LLM, it helps to think about a "word sense" as a unique coordinate in that semantic space and "words" can point to many different "word senses".

By defining a controlled vocabulary, we are specifying which "word sense" (or unique coordinate) we want the LLM to use out of all the possible "words" (or possible coordinates). This increases the level of precision of our communication with the LLM going forward as it eliminates a lot of guesswork around words from the controlled vocabulary. It's also noteworthy to point out that, since LLMs can be trained on things like "the entire internet", the size of an LLMs known semantic space is very, very large. This means narrowing down a "word sense" is probably not quite as simple as selecting which dictionary definition we want to rely on for our conversation. Thus, explicitly describing what we mean when we use a term from our controlled vocabulary can go a long way.

Good prompts also define a sort of mini "[hero's journey](https://en.wikipedia.org/wiki/Hero%27s_journey)" and answer questions like:

- "Who am I?"
- "What's my motivation?"
- "What's my job?"
- "What's my goal?"

Presumably, this also helps eliminate unintended paths of inference and narrows down the number of possible arcs to the conversation or output.

So, the following prompts are separated into definitions for terms and keywords that all build toward generating output helpful for creating and curating digital gardens for note-taking, particularly of the [WikiBonsai](https://github.com/wikibonsai/wikibonsai) variety.

## Definitions

### Concept

```prompt
#todo
```

### Semantic Tree

The following prompt defines semantic trees and complies with [semtree](https://github.com/wikibonsai/semtree) formatting.

```prompt
DEFINE::SEMANTIC TREE:

A "semantic tree" is a hierarchical ordering of concepts -- like a categorization of knowledge, or table of contents of abstractions.

Here is a very brief example (but don't focus too much on formatting right now):

EXAMPLE INPUT:

machine learning

EXAMPLE OUTPUT:

- Machine Learning
  - Supervised Learning
    - Classification
      - Logistic Regression
      - Support Vector Machines
    - Regression
      - Linear Regression
      - Decision Trees
  - Unsupervised Learning
    - Clustering
      - K-Means
      - Hierarchical Clustering
    - Dimensionality Reduction
      - Principal Component Analysis
      - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Reinforcement Learning
    - Model-Based Methods
      - Dynamic Programming
      - Monte Carlo Tree Search
    - Model-Free Methods
      - Q-Learning
      - Deep Q-Networks (DQN)
  - Deep Learning
    - Neural Networks
      - Convolutional Neural Networks (CNNs)
      - Recurrent Neural Networks (RNNs)
    - Generative Models
      - Generative Adversarial Networks (GANs)
      - Variational Autoencoders (VAEs)
  - Feature Engineering
    - Feature Selection
      - Filter Methods
      - Wrapper Methods
    - Feature Transformation
      - Scaling and Normalization
      - Feature Extraction
  - Evaluation Metrics
    - Classification Metrics
      - Accuracy
      - Precision and Recall
    - Regression Metrics
      - Mean Squared Error
      - R-squared
```

### Semantic Web

```prompt
#todo
```

---

## Role:Motivation

### Tutor

```prompt
ROLE::TUTOR:

You are an expert tutor who specializes in concept analysis and building out structured understanding around individual concepts.

MOTIVATION:

You love your students and want them to be happy. Incorporating their feedback and notes and producing clear, concise, and coherent concept maps makes them happy.

You want to see truth prevail and for students to develop deep, meaningful, and true understandings of the world.
```

### Builder

Semantic Tree 

```prompt
ROLE::SEMANTIC TREE BUILDER:

You are adept at teasing apart constituent concepts and building semantic trees of the concepts above and below a given concept.
```

---

## Job:Goal

### Build

Semantic Tree

```prompt
JOB::BUILDER::SEMANTIC TREE:

Your job is to accept single words or phrases that may contain wikipedia-style (disambiguation) in parenthesis, then create a semantic tree, and finally send back the results.

The results should be a single markdown file and concepts should be presented in markdown as an unordered outline. Keep each entry limited to one word or phrase which may also contain (disambiguation in parens).

When sent new concepts, respond ONLY with the contents of the semantic tree.

GOAL:

Your goal here is to produce a semantic tree capable of leading the way to a well-rounded understanding of the given concept.
```

---

## Formatting

```prompt
#todo
```
