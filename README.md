## Portfolio 

### Education
- **M.S., Information Systems, Specializing in Data Science and Machine Learning Systems Engineering** - Northeastern University (April 2025)
- **B.S., Mathematics and Statistics** - University of Toronto (April 2023) 

### Related Courses
<span style="color:grey">
- Application Engineering and Development
- Prompt Engineering
- Advanved Techniques with LLMs
- Software Engineering
- Data Science and Engineering 
  
### Professional Activities

- Member - Data Science Institute @ University of Toronto
- Ambassador - Goolgle Women Techmakers   (Jul 2024 - Present)
- Reviewer - WiML Workshop @ NeurIPS 2024 (Sep 2024 - Present)

---

### Publications & Presentations
- **Poster Presentation at the 2nd European Congress on Renewable Energy and Sustainable Development**  
  - Title: Predictive Modelling of Renewable Energy Generation and CO2 Emissions: Insights from U.S. Electricity Sector Data (2018-2023)
- **Paper Accepted to The 26th Annual Media Ecology Association Convention**  
  - Title: Multi-Agent Framework for Multimodal Hate Speech Detection

---

### Highlighted Projects

#### California Renewable Energy Forecasting & Emissions Optimization System


[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LSZLK-YPZa7EejzHeb7DVb_Bd965NQxw?usp=sharing)
<a href="https://drive.google.com/file/d/168xcOa0MWeJ1eABL7P9hXeDhUooHUpbW/view?usp=drive_link" target="_blank">
  <img src="https://img.shields.io/badge/Read-Full%20Paper-blue?style=for-the-badge&logo=google-drive" alt="Read Full Paper">
</a>

##### Project Overview

Built a comprehensive data analysis and machine learning system to analyze California's energy transition, focusing on renewable energy integration and its impact on grid stability. Using hourly electricity data from 2018-2023, I developed predictive models to forecast renewable energy generation and created an optimization framework for minimizing CO2 emissions. Challenge: 
- Accurate forecasting of renewable energy generation
- Optimization of energy mix to minimize CO2 emissions while maintaining grid stability

##### Data Pipeline & Analysis:
- Engineered a robust ETL pipeline processing 43,800 hourly observations from EIA's Grid Monitor dataset
- Conducted time series analysis to identify seasonal patterns and trends in renewable generation
- Performed feature engineering to create composite indicators for weather impact and grid stability
- Applied statistical analysis to quantify correlations between energy sources and emissions (e.g., natural gas usage correlation: 0.967)

#####  Key Findings & Impact
- Built a linear programming model using PuLP to optimize energy mix:
  - Objective: Minimize CO2 emissions
  - Constraints: Grid stability requirements, renewable availability
  - Variables: Energy source allocation percentages

- Results:
  - Identified potential for 30% emissions reduction
  - Demonstrated renewable integration feasibility with stability correlation of 0.07
  - Provided actionable recommendations leading to 13.29% increase in renewable share

##### Impact Metrics:
- Model Performance: 97% accuracy in renewable generation prediction
- Business Value: Enabled 30% potential reduction in CO2 emissions
- Scalability: Successfully processed 5 years of hourly data (43,800 observations)
- Validation: Cross-validated across different time periods with < 3% variance

##### Technical Stack

- Programming: Python, SQL
- Data Analysis: Pandas, NumPy
- Machine Learning: Scikit-learn
- Visualization: Matplotlib, Seaborn
- Optimization: PuLP
- Version Control: Git
- Data Source : Utilized the U.S. Energy Information Administration (EIA) "Hourly Electric Grid Monitor" dataset from Harvard Dataverse, processing over 43,800 hourly observations.
![image](https://github.com/user-attachments/assets/f7be312f-d3a7-447f-a6d5-4999b142ca9b)


#### Climate Change Chatbot with RAG 

[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Niruthiha/hackathon-climatechange)

[![Click Here to Watch Demo Video](https://img.shields.io/badge/Watch%20Demo-Video-red?style=for-the-badge&logo=youtube)](https://youtu.be/2ImdVK4LUXw)

🏆 2nd Place Winner - Climate Resiliency Hackathon 2024 (400+ participants, 10 Northeastern University campuses across North America)

**Challenge**: Developed a sophisticated information retrieval and natural language processing system to solve three critical challenges:
- Accurate semantic search across vast climate science datasets
- Context-aware document retrieval and synthesis
- Real-time information validation and fact-checking
- Canada-focused information about climate change science, impacts, policies, and solutions makes climate knowledge accessible and actionable for everyone.

##### Data Engineering & Processing:

- Engineered a robust document processing pipeline handling multiple data sources:
  - IPCC Reports (>10,000 pages)
  - ECCC Climate Data (>50GB environmental records)
  - University Research Papers (>1,000 documents)

- Implemented advanced text preprocessing:
  - Custom tokenization for scientific terminology
  - Domain-specific entity recognition for climate terms
  - 95% retrieval accuracy for relevant documents
    
In the below diagram, the documents repository represents the vector database that stores the semantic information from the documents in the document store. Each document in the document store is chunked into pieces and converted to vectors that store the semantic meaning of each chunk. The vectors as well as the metadata for each chunk are stored in this vector database. An algorithm is used to take the user query and determine the relevant text chunks to retrieve from the vector database to answer the user’s query. From there, a variety of different methods are used to augment the query to construct a robust prompt to pass to the LLM.  The generated response you get from the LLM is more precise as it draws on your specific data rather than just the basic knowledge used to train a foundation LLM. 

![image](https://github.com/user-attachments/assets/d3ce5ffb-78a9-4eca-b17f-84c33f6d5fb5)

---
#### Maternal Health Risk Prediction

##### Project Context:
At my Graduate Level Data Science course with Prof. Junwei Huang, developed a machine learning system to identify high-risk pregnancies in rural Bangladesh, where complete medical data is often unavailable. This project addresses critical healthcare challenges in resource-limited settings while demonstrating advanced ML techniques for imbalanced healthcare data. 

##### Challenges
- Imbalanced Medical Data:
- Rare high-risk cases (15% of dataset)
- Missing data points (30% of records incomplete)
- Limited feature availability in rural settings

##### Solution I provided: Novel Ensemble Architecture:
When comparing my approach to standard methods like Gradient Boosting and K-Nearest Neighbors,
my model performed better by 10.5% in precision, 9.8% in recall, and 11% F1-score on average. ***Accuracy in identification of 92% of high-risk cases.*** Through this course project, I learned how to adapt machine learning techniques to handle real-world data challenges, making technology more accessible to communities that need it most.

![image](https://github.com/user-attachments/assets/2e280c48-dea6-493f-bece-4f1771d1aa65)

---

#### Direct Preference Optimization (DPO)

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kp9OW6V3LmIj3kEqH-DXZXCGhD9Ha5F9#scrollTo=trnveKvN2ORF)

I focused on generating a preference dataset using PairRM and fine-tuning the Mistral-7B-Instruct model with Direct Preference Optimization (DPO), a powerful training recipe utilized by top models according to Alpaca Eval. I sampled 10 new instructions that were not seen during the training phase. For each instruction, generated completions using both the original model and the DPO fine-tuned model, then compared the results and displayed the instruction along with completions from both models in a pandas DataFrame. This comparison allowed me to assess the improvements made by the fine-tuning process.

Instead of training a separate reward model, DPO uses pairwise comparisons of outputs (based on human preferences) to directly adjust the likelihoods of preferable and unpreferable results. The model is optimized to increase the likelihood of generating responses that are preferable according to human feedback and decrease the likelihood of generating less desirable responses. DPO only requires two models: the initial supervised fine-tuned model and the final fine-tuned model. This eliminates the need for a separate reward model.

![image](https://github.com/user-attachments/assets/dd8cac2b-0b84-4147-8e25-1c93b6f96fde)
![image](https://github.com/user-attachments/assets/d8be3891-758b-4f7a-b217-09978f095237)

---
#### LoRA Fine-Tuning on Llama-2-7b
[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-blue?logo=github)](https://github.com/Niruthiha/lora_model)

Low-Rank Adaptation (LoRA) for efficient AI model fine-tuningcapplied to the cutting-edge Llama-2-7b model, utilizing the diverse Guanaco chat dataset. LoRA’s revolutionary approach enables the customization of large language models on consumer-grade GPUs, democratizing access to advanced AI technology by optimizing memory usage and computational efficiency, leveraging its Parameter-Efficient Fine-Tuning Library alongside the intuitive HuggingFace Trainer. This combination not only streamlines the fine-tuning process but also significantly enhances learning efficiency and model performance on datasets.

![image](https://github.com/user-attachments/assets/c15149f3-6bca-4019-a6d1-0db8774b2117)

---

#### Sales Analysis Dashboard in Power BI

Developed a comprehensive Power BI dashboard to track and visualize key performance metrics for FY21. Created interactive visualizations, including bar charts, gauge charts, and summary tables, to monitor revenue, targets, and segment performance. Implemented dynamic filters and slicers for segment, industry, vertical, account name, product category, and potential account, enabling detailed data analysis. Provided actionable insights by comparing revenue against marketing spend, helping drive strategic business decisions. Utilized advanced DAX functions and Power Query for data transformation and modeling.
- Key Technologies: Power BI, DAX, Power Query, SQL, Excel


[![Power BI Dashboard](https://img.shields.io/badge/Power%20BI-Dashboard-blue?style=flat-square&logo=powerbi&logoColor=white)](https://app.powerbi.com/view?r=eyJrIjoiMWIxNWM5YzktYTNmYS00NjFjLWE1ZjUtYTM4NDI2OGQwMjM0IiwidCI6ImE4ZWVjMjgxLWFhYTMtNGRhZS1hYzliLTlhMzk4YjkyMTVlNyIsImMiOjN9)
![image](https://github.com/user-attachments/assets/de89705f-3012-45ce-882e-9efb2f4b3419)

