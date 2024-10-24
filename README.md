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
![image](https://github.com/user-attachments/assets/506209cd-59f4-4892-99da-0771baa15f73)


#### Climate Change Chatbot with RAG 
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://m9c2pz0j-8501.use.devtunnels.ms/)
[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-blue?logo=github)](https://github.com/Niruthiha/Chatbot-RAG-Vector-Database)
[![Click Here to Watch Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-red)](https://youtu.be/Xx-mtNaRHeI)

The Climate Change Chatbot is an interactive tool designed to provide users with information about climate change using advanced natural language processing techniques. By leveraging a Retrieval-Augmented Generation (RAG) approach, this chatbot utilizes a combination of a vector database and OpenAI's language model to deliver accurate and relevant responses based on user queries. Utilizing the GPT-3.5-turbo model, the chatbot generates context-aware responses based on the queried information. The data used for the Climate Change Chatbot is derived from the IPCC AR6 WGII Technical Summary (IPCC_AR6_WGII_TechnicalSummary.pdf). This technical summary complements and expands on the key findings of the Working Group II contribution to the Sixth Assessment Report (AR6), providing essential insights into climate change impacts and responses. The retrieval process is what makes RAG unique. 

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

<span style="color:grey">In this project, I utilized various machine learning models such as Linear Regression, Decision Trees, Random Forest, k-Nearest Neighbors (k-NN) to predict flight prices. This repository includes the code, methodologies, and detailed explanations for each step of the project. The project involved splitting the dataset into training and test sets using train_test_split from sklearn.model_selection.

Initially, I built a basic random model, then iteratively improved it using hyperparameter tuning. A RandomForestRegressor was trained on the data, and performance metrics such as R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) were calculated. The model was then saved using pickle for future predictions.

To automate the machine learning pipeline, custom evaluation metrics were defined, and the process was streamlined to provide comprehensive results including training score, predictions, and various error metrics. Hyperparameter tuning was conducted using RandomizedSearchCV to optimize the model parameters.

This repository includes the code, methodologies, and detailed explanations for each step of the project, providing insights into the entire machine learning workflow from data preprocessing to model evaluation and deployment

![Screenshot 2024-07-05 114600](https://github.com/Niruthiha/portfolio/assets/157150830/c21b1e82-34ae-47c7-9c46-6b8d6fdf6572)


---
#### Sales Analysis Dashboard in Power BI

Developed a comprehensive Power BI dashboard to track and visualize key performance metrics for FY21. Created interactive visualizations, including bar charts, gauge charts, and summary tables, to monitor revenue, targets, and segment performance. Implemented dynamic filters and slicers for segment, industry, vertical, account name, product category, and potential account, enabling detailed data analysis. Provided actionable insights by comparing revenue against marketing spend, helping drive strategic business decisions. Utilized advanced DAX functions and Power Query for data transformation and modeling.
- Key Technologies: Power BI, DAX, Power Query, SQL, Excel


[![Power BI Dashboard](https://img.shields.io/badge/Power%20BI-Dashboard-blue?style=flat-square&logo=powerbi&logoColor=white)](https://app.powerbi.com/view?r=eyJrIjoiMWIxNWM5YzktYTNmYS00NjFjLWE1ZjUtYTM4NDI2OGQwMjM0IiwidCI6ImE4ZWVjMjgxLWFhYTMtNGRhZS1hYzliLTlhMzk4YjkyMTVlNyIsImMiOjN9)
![image](https://github.com/user-attachments/assets/de89705f-3012-45ce-882e-9efb2f4b3419)

