## Portfolio 

## About Me
Doctoral researcher specializing in AI/ML applications for Machine Learning and Software Engineering

## Education
- **Doctor of Philosophy - PhD, Engineering** - École de technologie supérieure (Year 1)
  - Supervisors: Dr. Manel Abdellatif and Dr. Taher Ghaleb
- **M.S., Information Systems, Data Science and ML Systems Engineering** - Northeastern University (April 2025)
- **B.S., Mathematics and Statistics** - University of Toronto (April 2023)

### Related Courses
<span style="color:grey">
- Application Engineering and Development
- Prompt Engineering
- Advanved Techniques with LLMs
- Software Engineering
- Data Science and Engineering 
- Advanced Data Sci/Architecture
- Data Management and Database Design

### Professional Experience

- **Machine Learning Engineer** - Vector Institute for Artificial Intelligence
- **Graduate Teaching Assistant** - Northeastern University

### Professional Activities

- Member - Data Science Institute @ University of Toronto
- Ambassador - Goolgle Women Techmakers   (Jul 2024 - Present)
---

### Publications & Presentations
- **Paper Accepted to 2025 IEEE 13th International Conference on Healthcare Informatics (ICHI)**
  - Title: Multidimensional Analysis of Specific Language Impairment Using Unsupervised Learning Through PCA and Clustering
- **Poster Presentation at the 2nd European Congress on Renewable Energy and Sustainable Development**  
  - Title: Predictive Modelling of Renewable Energy Generation and CO2 Emissions: Insights from U.S. Electricity Sector Data (2018-2023)
---

### Highlighted Projects

#### Named Entity Recognition for Restaurant Search Queries

<!-- Open in GitHub Badge -->
[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Niruthiha/DistilBERT-based-NER-/tree/main)

<a href="https://huggingface.co/niruthiha/restaurant-ner" target="_blank">
  <p>View The Model in Hugging Face (800+ model downloads as of June 2025)</p>
</a>
<!-- Full Paper Link -->
<a href="https://drive.google.com/file/d/1024OmUDdoOz9B_ZG8JTbiwPnMdmSaFM8/view?usp=sharing" target="_blank">
  <img src="https://img.shields.io/badge/Read-Full%20Paper-blue?style=for-the-badge&logo=google-drive" alt="Read Full Paper">
</a>

This project showcases my expertise in Natural Language Processing and model deployment by developing a Named Entity Recognition (NER) system specifically designed to understand restaurant search queries.  Leveraging the power of transfer learning, I fine-tuned a DistilBERT model to accurately extract key structured information like restaurant ratings, cuisines, locations, and amenities from free-form text.  This model excels at transforming natural language input into actionable data, demonstrating real-world applicability for enhancing search and recommendation systems. Key highlights include:

- Task & Model: Developed a Token Classification model using DistilBERT for Named Entity Recognition (NER).
- Domain Expertise: Focused on the restaurant search domain, addressing a practical application of NLP.
- Performance Metrics: Achieved strong evaluation metrics on the test set, including a Precision of 0.766, Recall of 0.803, F1-Score of 0.784, and Accuracy of 0.916, demonstrating robust model performance.
- Dataset & Methodology: Trained and evaluated on the MIT Restaurant Search NER dataset, employing effective data preprocessing and tokenization strategies for subword alignment.
- Deployment & Accessibility: Successfully deployed and hosted the fine-tuned model on the Hugging Face Model Hub, making it readily accessible for inference and further use by the community.
- Potential Impact: Demonstrates the ability to build components for real-world applications such as intelligent restaurant search engines, food delivery platforms, and conversational AI assistants for dining recommendations.

This project not only highlights my skills in NLP and deep learning but also demonstrates my ability to tackle domain-specific problems and deploy models for practical use.  
![Screenshot 2025-03-10 185601](https://github.com/user-attachments/assets/43ed56f8-b426-4ad9-bf7e-5de77677c848)
![Screenshot 2025-03-10 201215](https://github.com/user-attachments/assets/af823ccd-6b28-47ed-a541-5b3690cced0b)



---
####  Card Fraud Detection: Anomaly Detection
<a href="https://drive.google.com/file/d/1Tf296gT8k77i_R57fObik7Ek-vuwzdpZ/view?usp=sharing" target="_blank">
  <img src="https://img.shields.io/badge/Read-Full%20Paper-blue?style=for-the-badge&logo=google-drive" alt="Read Full Paper">
</a>

##### The Challenge: 
Built an AI system to detect credit card fraud in highly imbalanced data where only 0.17% of 284,000+ transactions were fraudulent - a critical problem costing financial institutions billions annually.
##### My Solution: 
Developed an advanced Random Forest model with SMOTE oversampling technique to balance the dataset and improve fraud detection accuracy.
Impact & Results:

##### 84% Precision - Reduced false alarms by 13.5%, saving operational costs
##### 80% Recall - Detected 14% more fraudulent transactions, preventing financial losses
##### PR-AUC - 89.2% I used Precision-Recall AUC instead of ROC-AUC because with such extreme class imbalance (0.17% fraud), ROC can be misleadingly optimistic. PR-AUC better reflects performance on the critical minority class.

##### Why This Matters: 
Financial fraud detection is a $32 billion market with direct business impact. This project demonstrates my ability to solve complex, imbalanced data problems that are common in finance, healthcare, and security - delivering measurable ROI through improved accuracy and reduced operational costs.
##### Technologies: Python, Scikit-learn, Random Forest, SMOTE, Statistical Analysis

![image](https://github.com/user-attachments/assets/749ab024-b7a2-4118-9587-e58827ed548e)


---
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

#### MAHD: A Conservative Multi-Agent System for Contextual Hateful Meme Detection Using GPT-4

<a href="https://drive.google.com/file/d/1cWR93vSgEHNEh4futLYy68c68_c-ZEi6/view?usp=drive_link" target="_blank">
  <img src="https://img.shields.io/badge/Read-Full%20Paper-blue?style=for-the-badge&logo=google-drive" alt="Read Full Paper">
</a>


##### Project Overview
MAHD (Multi-Agent Hate Detection) is a novel dual-agent system for robust hateful meme detection in social media content. Built on GPT-4, MAHD employs a conservative classification approach that achieves high precision while effectively capturing subtle forms of harmful content

##### Key Features

🤖 Dual-agent architecture for comprehensive content analysis
🎯 Conservative classification protocol with strict calibration
📊 Adaptive reasoning system based on content complexity
🔄 Probabilistic decision framework using Bayesian inference
📝 Detailed explanation generation for moderation decisions

##### Performance Metrics

81.5% Overall Accuracy
93.02% Recall Rate
94% Accuracy in Explicit Hate Speech Detection
91% Accuracy in Identifying Calls to Violence

![image](https://github.com/user-attachments/assets/56ca7ee4-0d51-4706-ad49-226fbed34e05)

![image](https://github.com/user-attachments/assets/a2cc01d6-b70f-459d-ba15-b7f10c372c13)

---

#### Multivariate Analysis of Language Impairment Patterns Using Principal Component Analysis and Clustering

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/112RXS5nncvShS2bizJdYzrrO0p0BGCfS/view?usp=sharing)

<a href="https://drive.google.com/file/d/1HUCsf3MNkyTJl8rLab1kTe2Mbh52-yLL/view?usp=sharing" target="_blank">
  <img src="https://img.shields.io/badge/Read-Full%20Paper-blue?style=for-the-badge&logo=google-drive" alt="Read Full Paper">
</a>

##### Project Overview
This project applies advanced data science techniques to analyze patterns in language impairment, utilizing a dataset of 1,163 participants with 64 linguistic features.This project explores patterns in language development data using advanced dimensionality reduction and unsupervised learning techniques. By applying Principal Component Analysis (PCA) to a comprehensive dataset of language metrics, I reduced the high-dimensional data to its most informative components, with the first three components explaining 48.46% of the total variance. Using K-means clustering on these principal components revealed two distinct natural groupings in language development patterns. The analysis demonstrated robust cluster formation, particularly in PC1-PC2 space, with strong silhouette scores (0.380-0.460) indicating well-defined clusters. Notably, the clustering patterns showed remarkable consistency (96.5% agreement) between different PC space combinations, suggesting stable underlying structures in language development. The project implements various evaluation metrics and visualizations, including 3D representations and boundary analysis, to thoroughly validate the clustering results. These findings contribute to our understanding of natural language development patterns and demonstrate the value of unsupervised learning in discovering underlying structures in developmental data that could improve early diagnosis of language disorders.

##### Key Data Science Components:

- Implemented Principal Component Analysis (PCA) to reduce 64 dimensions to 14 significant components, explaining 83.55% of variance
- Applied K-means clustering to identify distinct language profiles in high-dimensional space
- Performed feature importance analysis to identify key linguistic markers
- Conducted statistical validation using silhouette scores and variance explained ratios
- Developed visualization techniques for high-dimensional linguistic data

![image](https://github.com/user-attachments/assets/369a5135-9e2e-4ecb-8e27-7bb12784a796)

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
#### Web Scraping Project: Financial Data Collection from Yahoo Finance

Technologies Used: Python, BeautifulSoup4, Pandas, Requests

Description:
Developed a web scraping system to automate the collection of financial metrics from Yahoo Finance, focusing on major **S&P500** publicly traded companies like Apple, Google, and Microsoft. The project targeted the Statistics pages of companies, gathering structured data such as balance sheets, income statements, cash flow statements, and management effectiveness metrics.

Key Features:

- Automated Data Collection: Designed functions to parse HTML, construct dynamic URLs, and handle rate-limiting challenges.
- Data Processing Pipeline: Organized data into structured formats (CSV) for analysis, using custom scripts for data flattening and validation.
- Error Handling & Reliability: Implemented robust error tracking, request management, and data validation mechanisms.
- Sample Output: Financial metrics like Return on Equity (ROE), Return on Assets (ROA), Profit Margins, Gross Margins, Operating Margins, Total Revenue, Year-over-Year (YoY) Revenue Growth, Operating Cash Flow, Free Cash Flow, Debt-to-Equity Ratio, Current Ratio, Quick Ratio, Price-to-Earnings (P/E) Ratio, Price-to-Sales (P/S) Ratio, Market Capitalization, Historical Revenue, Net Income, and Earnings Per Share (EPS).

![image](https://github.com/user-attachments/assets/3238edb8-e530-41b8-b476-3cb911cbe9fb)

---

#### Sales Analysis Dashboard in Power BI

Developed a comprehensive Power BI dashboard to track and visualize key performance metrics for FY21. Created interactive visualizations, including bar charts, gauge charts, and summary tables, to monitor revenue, targets, and segment performance. Implemented dynamic filters and slicers for segment, industry, vertical, account name, product category, and potential account, enabling detailed data analysis. Provided actionable insights by comparing revenue against marketing spend, helping drive strategic business decisions. Utilized advanced DAX functions and Power Query for data transformation and modeling.
- Key Technologies: Power BI, DAX, Power Query, SQL, Excel


[![Power BI Dashboard](https://img.shields.io/badge/Power%20BI-Dashboard-blue?style=flat-square&logo=powerbi&logoColor=white)](https://app.powerbi.com/view?r=eyJrIjoiMWIxNWM5YzktYTNmYS00NjFjLWE1ZjUtYTM4NDI2OGQwMjM0IiwidCI6ImE4ZWVjMjgxLWFhYTMtNGRhZS1hYzliLTlhMzk4YjkyMTVlNyIsImMiOjN9)
![image](https://github.com/user-attachments/assets/de89705f-3012-45ce-882e-9efb2f4b3419)

