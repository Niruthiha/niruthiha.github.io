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

#### Member - Data Science Institute @ University of Toronto
#### Ambassador - Goolgle Women Techmakers   (Jul 2024 - Present)
- Empowering women in tech by organizing events, speaking at conferences, creating educational content, and mentoring young professionals.
- Actively engaged in building an inclusive tech community, promoting gender diversity, and encouraging female participation in STEM fields through outreach and leadership.

#### Reviewer - WiML Workshop @ NeurIPS 2024 (Sep 2024 - Present)
- Reviewed and evaluated machine learning research abstracts for the WiML 2024 Workshop.
- Assessed submissions based on criteria such as relevance, clarity, novelty, and technical quality.
- Provided detailed, constructive feedback to authors to help improve the quality of their research contributions.


### Highlighted Projects

#### Predicting Renewable Energy Generation and C02 Emissions in California

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LSZLK-YPZa7EejzHeb7DVb_Bd965NQxw?usp=sharing)
<a href="https://drive.google.com/file/d/168xcOa0MWeJ1eABL7P9hXeDhUooHUpbW/view?usp=drive_link" target="_blank">
  <img src="https://img.shields.io/badge/Read-Full%20Paper-blue?style=for-the-badge&logo=google-drive" alt="Read Full Paper">
</a>



![image](https://github.com/user-attachments/assets/f7be312f-d3a7-447f-a6d5-4999b142ca9b)

This project analyzes California's energy mix and develops machine learning models to predict renewable energy generation using hourly electricity data from 2018 to 2023.

##### Key Findings

###### 1. Renewable Energy Trends
- Renewable energy usage in California increased from 32.63% in 2018 to 45.92% in 2023.
- Solar energy showed the most significant growth, rising from 14.09% to 19.80% of the energy mix.
- Wind energy also increased, from 7.75% to 9.75%.

###### 2. Energy Mix Evolution
- Natural gas remains the primary energy source but decreased from 51.26% in 2018 to 41.78% in 2023.
- Coal usage significantly declined from 5.04% to 1.22%.
- Hydroelectric power showed high variability, ranging from 5.70% to 17.45% depending on the year.

###### 3. CO2 Emissions
- CO2 emissions intensity decreased from 0.576 lbs/kWh in 2018 to 0.403 lbs/kWh in 2023.
- Strong positive correlation between natural gas usage and CO2 emissions (0.967).
- Strong negative correlation between solar energy and CO2 emissions (-0.737).

###### 4. Predictive Modeling
- Random Forest model performed best in predicting renewable energy generation (R² = 0.97).
- Most important features for prediction: Renewable Percentage, Hour, Grid Stability.

###### 5. Grid Stability
- Weak positive correlation (0.07) between renewable energy percentage and grid stability.
- Suggests California's grid is managing increased renewable integration without significant stability issues.

###### 6. Optimization Model
- Linear programming model developed to optimize energy mix for minimal CO2 emissions.
- Suggested increasing nuclear, hydro, and wind while reducing natural gas for optimal low-emission mix.

###### Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PuLP

###### Data Source
U.S. Energy Information Administration (EIA) "Hourly Electric Grid Monitor" dataset, accessed through Harvard Dataverse.


#### Climate Change Chatbot with RAG 
[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Niruthiha/hackathon-climatechange)

[![Click Here to Watch Demo Video](https://img.shields.io/badge/Watch%20Demo-Video-red?style=for-the-badge&logo=youtube)](https://youtu.be/2ImdVK4LUXw)


ClimateConnect is an innovative chatbot that provides accurate, Canada-focused information about climate change science, impacts, policies, and solutions. By leveraging cutting-edge AI and authoritative data sources, ClimateConnect makes climate knowledge accessible and actionable for everyone.

Retrieval-augmented generation (RAG) architecture for dynamic, context-aware question answering
Integration of trusted data sources: 
- Environment and Climate Change Canada (ECCC)
- IPCC (Intergovernmental Panel on Climate Change) Reports, including the 2023 Synthesis Report
- University of Manitoba climate research and expertise

In the below diagram, the documents repository represents the vector database that stores the semantic information from the documents in the document store. Each document in the document store is chunked into pieces and converted to vectors that store the semantic meaning of each chunk. The vectors as well as the metadata for each chunk are stored in this vector database. An algorithm is used to take the user query and determine the relevant text chunks to retrieve from the vector database to answer the user’s query. From there, a variety of different methods are used to augment the query to construct a robust prompt to pass to the LLM.  The generated response you get from the LLM is more precise as it draws on your specific data rather than just the basic knowledge used to train a foundation LLM. 

![image](https://github.com/user-attachments/assets/d3ce5ffb-78a9-4eca-b17f-84c33f6d5fb5)

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
#### Flight Price Prediction

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YwIu4swMWVN6OUnK6khOYhvDQ5ZGHrs_?usp=sharing)

<span style="color:grey">
At my Graduate Level Data Science course with Prof. Junwei
Huang, I worked on a project using healthcare data from rural Bangladesh to explore how machine learning could
improve maternal healthcare in resource-limited settings. I built a system to identify high-risk pregnancies in places
where complete medical data is not always available. For this project, I implemented a novel ensemble learning
approach combining Decision Trees, Random Forests, and Gradient Boosting algorithms. The real challenge was
handling imbalanced data distribution in medical cases - a common issue in healthcare datasets. I overcame this by
applying the Synthetic Minority Over-sampling Technique to balance the class distribution and sample weighting
to give higher importance to misclassified instances, further addressing the class imbalance issue. The combination
of these enabled the algorithm to effectively handle the imbalanced data and improve the identification of high-risk
pregnancies. When comparing my approach to standard methods like Gradient Boosting and K-Nearest Neighbors,
my model performed better by 10.5% in precision, 9.8% in recall, and 11% F1-score on average. Through this
course project, I learned how to adapt machine learning techniques to handle real-world data challenges. Working
with healthcare data taught me the importance of building inclusive solutions that can work effectively even with
incomplete information, making technology more accessible to communities that need it most

![image](https://github.com/user-attachments/assets/2e280c48-dea6-493f-bece-4f1771d1aa65)

---
#### Sales Analysis Dashboard in Power BI

Developed a comprehensive Power BI dashboard to track and visualize key performance metrics for FY21. Created interactive visualizations, including bar charts, gauge charts, and summary tables, to monitor revenue, targets, and segment performance. Implemented dynamic filters and slicers for segment, industry, vertical, account name, product category, and potential account, enabling detailed data analysis. Provided actionable insights by comparing revenue against marketing spend, helping drive strategic business decisions. Utilized advanced DAX functions and Power Query for data transformation and modeling.
- Key Technologies: Power BI, DAX, Power Query, SQL, Excel


[![Power BI Dashboard](https://img.shields.io/badge/Power%20BI-Dashboard-blue?style=flat-square&logo=powerbi&logoColor=white)](https://app.powerbi.com/view?r=eyJrIjoiMWIxNWM5YzktYTNmYS00NjFjLWE1ZjUtYTM4NDI2OGQwMjM0IiwidCI6ImE4ZWVjMjgxLWFhYTMtNGRhZS1hYzliLTlhMzk4YjkyMTVlNyIsImMiOjN9)
![image](https://github.com/user-attachments/assets/de89705f-3012-45ce-882e-9efb2f4b3419)

