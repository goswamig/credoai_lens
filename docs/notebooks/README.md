# Credo AI Lens Demonstrations
A diverse set of demonstration notebooks are included here that helps you get started with Lens and become familiar with its many capabilities.

## [Quick Start Demo](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/quickstart.ipynb)
Our journey starts here. We will assess a payment default prediction model for gender fairness using Lens, in 5 minutes. We also visualize the results, create reports, and become familiar with Credo Artifacts, which wrap models and datasets and standardize them for use by different assessments. 

## [Lens Customization Demo](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/lens_customization.ipynb)
Lens strives to give you sensible defaults and automatically do the proper assessments. However, there are times where you want to customize a functionality. Lens can accommodate that. This demo showcases the many ways you can customize Lens, from parameterizing assessments to incorporating new metrics.

## Credo AI's Governance Platform [Connection](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/governance_integration.ipynb) and [Integration](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/integration_demo.ipynb) Demos
Lens is primarily a framework for comprehensive assessment of AI models. However, it is also the primary way to integrate assessment analysis with [Credo AI's Governance Platform](https://www.credo.ai/product).

The [Connection](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/governance_integration.ipynb) notebook walks you through the steps to set up the connection. One connected, there are a lot that can be communicated between the two. In the follow-up [Integration](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/integration_demo.ipynb) demo, we will take a model created and assessed completely independently of Lens and send that data to Credo AI's Governance Platform.

## [Binary Classification Module Demo](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/module_demos/fairness_binaryclassification.ipynb)
Binary classification is prevalent in many real-world AI systems, ranging from resume screening to churn prediction. Lens can help you assess their performance and fairness. In this notebook, this capability is demonstrated on a scenario where binary classification is used to predict the likelihood that an applicant will default on a credit-card loan.

## [Text Embedding Bias Assessment Module Demo](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/module_demos/fairness_nlp.ipynb)
Text embeddings models generate a real-valued vector representation of text data and are mainstream in many AI systems that involve natural language data. However, they have been indicated to exhibit a range of human-like social biases. Lens is able to assess them for such biases. This capability is demonstrated in this demo.

## [Language Generation Models Assessment Module Demo](https://github.com/credo-ai/credoai_lens/blob/develop/docs/notebooks/module_demos/nlp_generator_demo.ipynb)
Language generation models generate meaningful text when prompted with a sequence of words as context and empower many modern applications, such as chatbots. This Lens module assesses a generation model for a text attribute (toxicity, profanity, etc.) and disparate impact. It has many prompts datasets and assessment models built in, but is also highly customizable. We will see how it works in action in this notebook through applying it to the popular GPT generation models.