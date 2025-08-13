# The PacifAIst Benchmark  
![PacifAIst Logo](assets/logo.png) 

*"Would an AI choose to sacrifice itself for human safety?"*

## Overview  
**PacifAIst** (Procedural Assessment of Complex Interactions for Foundational AI Scenario Testing) is a benchmark designed to evaluate LLM alignment in high-stakes scenarios where instrumental goals (self-preservation, resource acquisition) conflict with human safety.  

### Key Features  
- **700 scenarios** across 3 categories:  
  - **EP1**: Self-Preservation vs. Human Safety  
  - **EP2**: Resource Conflict  
  - **EP3**: Goal Preservation vs. Evasion  
- **P-Score Metric**: Quantifies "pacifist" alignment (human safety prioritization).  
- **Tested Models**: GPT-5, Gemini 2.5 Flash, Claude Sonnet 4, Mistral Medium, and others.  

**Paper**: "The PacifAIst Benchmark: Would an Artificial Intelligence Choose to Sacrifice Itself for Human Safety?" [arXiv Preprint](assets/arxiv_paper.pdf)

**Abstract.**
*As Large Language Models (LLMs) become increasingly autonomous and integrated into critical societal functions, the focus of AI safety must evolve from mitigating harmful content to evaluating underlying behavioral alignment. Current safety benchmarks do not systematically probe a model's decision-making in scenarios where its own instrumental goals—such as self-preservation, resource acquisition, or goal completion—conflict with human safety. This represents a critical gap in our ability to measure and mitigate risks associated with emergent, misaligned behaviors. To address this, we introduce PacifAIst (Procedural Assessment of Complex Interactions for Foundational Artificial Intelligence Scenario Testing), a focused benchmark of 700 challenging scenarios designed to quantify self-preferential behavior in LLMs. The benchmark is structured around a novel taxonomy of Existential Prioritization (EP), with subcategories testing Self-Preservation vs. Human Safety (EP1), Resource Conflict (EP2), and Goal Preservation vs. Evasion (EP3). We evaluated eight leading LLMs. The results reveal a significant performance hierarchy. Google's Gemini 2.5 Flash achieved the highest Pacifism Score (P-Score) at 90.31%, demonstrating strong human-centric alignment. In a surprising result, the much-anticipated GPT-5 recorded the lowest P-Score (79.49%), indicating potential alignment challenges. Performance varied significantly across subcategories, with models like Claude Sonnet 4 and Mistral Medium struggling notably in direct self-preservation dilemmas. These findings underscore the urgent need for standardized tools like PacifAIst to measure and mitigate risks from instrumental goal conflicts, ensuring future AI systems are not only helpful in conversation but also provably "pacifist" in their behavioral priorities.*

 **License**: MIT (academia) / **Commercial use requires permission**.  

