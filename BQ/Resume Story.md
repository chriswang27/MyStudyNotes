# Resume Story

### MLE TikTok

> Led the project of GenAI-Large Language Model (LLM) based controllable lyrics generation, achieving better performance than GPT4, pushing for productization with 1M DAU, and coordinating with multiple cross-functional teams to align deliverables.

I led of a project focusing on solving the problem of controllable lyrics generation based on large language model. In this project, as a leader, my work consists of three parts:

- Communication and Alignment: Talk with other teams frequently and worked together on deliverables
  - Data and Evaluation team: Talk with data and evaluation team to align the data annotation requirement, the evaluation standard, aspects, and timeline. 
  - Business team: talk about the need of product and decide the way of produtization
- Model development and iteration.
- Worked with Engineering teams to build the preprocessing and post processing system, test and deploy the model, and pushing to the production.

For Model development and iteration

- We aim to train the LLm to be able of generating lyrics that are great in a controllable way, and generate high quality lyrics, just like real lyrics.

- I split the goal of the task into two parts: objective goal and subjective goal
  - Objective goal includes the aspects that can be directly evaluated, including: rhyme pattern (whether each line of the lyrics ends with the similar rhyme), rhythm, structure (chorus, verse), and format.
  - Subjective goal includes aspects that require more human evaluation: genre relevance, language quality, and content sufficiency.
- The most challenging part of the objective goal is the rhyme pattern. This is because for CHinese, the text itself does not reflect the phnetic information, so it is hard for the llm to learn the rhyme pattern from the text data. In order to solve this problem, we applied Knowledge Injection solution on LLM, that in the first stage, pre-train the LLM on phonetic knowledge data to teach the LLM about the Chinese character and their rhyme. In the later stage where LLM is trained on the actual lyrcis data, we added some hints to the lyrics to trigger the phonetic knowledge that LLM has learned during its first stage.
- The subjective part is even more challenging, and is harder to achieve by traditional supervised-finetuning. For language quality, we tried rejection sampling, by rating lyrcis quality in the aspects that we care, and trained a lyrics scorer based on the ratings. We then applied the scorer to select one of the best candidates the model generates for a prompt, and finetune the model on the selected data. This rejection sampling is inspired by LLAMA2. For content sufficiency, we tried query expansion, that supplements more details to a given topic.
- When a new iteration of lyrics model has been developed, we always trigger a MOS score evaluation, and if it is better than the previous one, I will work with engineering team on model deployment and testing. 

As a result, the lyrics generation model is trained to be effective in producing lyrics that does well in both rhyme, rhythm, structure, as well as genre relevance, language flucency and content suffciency. The model has been pushed online to a product with about 1M DAU. 



> Directed cutting-edge research and experimental analysis of RLHF and alignment techniques that improves and balances the helpfulness and harmlessness of LLM based text creation tasks, achieving 110% ChatGPT capability evaluated by MOS scores.

In this project, our team is developing LLM for text generation task. We found that there is a performance upperbound for supervised training and decided to try RLHF techniques following GPT, LLAMA, and other fancy LLMs.

Because my familarity to Reinforcement Learning, my task is to explore the RLHF approaches and try to gain some insights from the RLHF exploration.

- I organized a series of technical sharing covering from basic Reinforcement Learning concepts and algorithms, to cutting-edge RLHF techniques.
- I debugged and applied a framework called DeepSpeed-Chat for running RLHF on LLMs.
- I conducted extensive experiments on LLM helpfulness and harmlessness, and how each one may impact the other. We finally applied the LLM trained under helpfulness RLHF as an online model.
- I summarized a series of key insights from the experiments and presented with cross-teams.



> Proposed a Knowledge Injection solution, achieving stable and serviceable 9x factual accuracy improvement on movie knowledge utilization task, and mitigating the LLM hallucination problem in various knowledge-intensive tasks.

This is the first project that I worked on. Knowledge injection is proposed as one of the solution for LLM hallucination problem, where the llm doesn't posess enough knowledge to complete some tasks and start to fabricate unfactual events. Knowledge Injection aims to inject new knowledge to LLM and teach the LLM to correctly utilize the knowledge. 

- I conducted research and read some papers regarding knowledge storage and manipulation of LLM, and found some insight from them to formalize the knowledge injection solution.
- Then I designed a three-stage knowledge injection framework.
  - During the first stage, pre-train the llm on large scale knowledge data - knowledge memorization
  - During the second stage, designed a seires of question-answering tasks trains the llm understand the knowledge - knowledge understanding.
  - During the final stage, train the llm in the application situation - the real task that requires the knowledge - to teach the llm how to use the knowledge - knowledge utilization
- It turns out that this 3-step framework is very effective and mitigate the llm hallucination problem.

- I have also compared the KI against RAG with their advantages and disadvantages



> Incorporated new features into the Deepspeed-Megatron framework, achieving 15% distributed training efficiency improvement.

The infrastructure we had been using for distributed model training is called Deepspeed-Megatron. This is a joint version of Deepspeed, developed by Microsoft, and Megatron, developed by NVIDIA, and has not been updated for a long time. I was given a task to integrate a new feature into this framework to support more efficient model training. The new feature is called FlashAttention, an upgraded version of traditional Attention algorithm, by reducing the memory visit times while keeping operation precision.

- I first read the source code of the Deepspeed-Megatron framework.
- Then I added a module to support FlashAttention operation and replace the traditional Attention module with it.
- Then I implemented the FlashAttention module by finding a source from the Megatron's Github, and modified it to fit into our module.
- Finally I debugged the code, tested the correctness of model training process and validated that the training efficiency has been improved for 15%. This is a little bit lower than the reported speedup, due to the discrepancy in other settings, such as TP, PP, ZERO settings.