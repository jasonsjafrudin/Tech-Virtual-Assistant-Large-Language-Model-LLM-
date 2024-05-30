# Tech-Virtual-Assistant-Large-Language-Model
Engineered Large Language Model and Web Application for streamlining Market Research in Tech Industry. The goal is to address the increasing difficulties of keeping pace with rapid changes in market conditions and consumer preferences in the tech industry by leveraging advanced machine learning techniques to build and deploy a LLM and application that transforms the quality and timeliness of conducting market research in this dynamic sector.

## Data Extraction and Preprocessing:
<img width="403" alt="image" src="https://github.com/jasonsjafrudin/Tech-Virtual-Assistant-Large-Language-Model-LLM-/assets/61297201/8bfd8f7f-3798-4fbe-8b2e-ab0acb2dfb1d">

Original data is collected from RSS News Feed in json files written in HTML format. Each json file represents an unique article, and the json file also contains the original url, date of publication, author information, and html content of the article. We have filtered data specific to tech related articles (around 2.1 Gb in size). Utilized beautifulsoup package to transform data into a human-readable text from the HTML text, extracting pertinent information such as article titles, authors, publication dates, and content. I also defined a cleaning function to deal with the dirty news article texts; eliminating newlines, tabs, remnants of web crawls, and other irrelevant text with the help of regex module. 


## Methodology Approaches:
<img width="534" alt="image" src="https://github.com/jasonsjafrudin/Tech-Virtual-Assistant-Large-Language-Model-LLM-/assets/61297201/bbde32ca-8903-4df3-adea-aaa6ba38b694">

LLM for text summarization- Unsupervised approach. Fine-tuned open source BART LLM for text summarization using my proprietary tech news articles dataset (unlabeled). This initial training phase will involve tokenizing and feeding our dataset into the model architecture, allowing it to learn the intricacies, nuances, and concepts within tech related articles through self-supervised learning objectives. I will also fine-tune the model through adjusting various hyperparameters and optimizing the model's parameters to ensure efficient understanding and performance specifically to our unique requirements and objectives. 
LLM for NER and Sentiment Analysis- Supervised approach. I first generated labeled datasets through custom NLP models. 
First is a entity Identification System to accurately identifying entities mentioned in the tech news articles. The methodology I used is an open-source NLP library, spaCy. It is a neural network-based NER model to identify and extract entities, including product, organization, person, and location. By processing our tech news articles through spaCy's NER pipeline, it will annotate our tech news articles with entity labels.
Second is a sentiment analysis NLP model utilizing the state of the art NLP library developed by Zalando Research, Flair. It offers pre-trained sentiment analysis models that can classify and label the sentiment (positive, negative, neutral) of our tech news articles.
These labeled datasets are then used to fine-tune and train open source BERT and DistillBERT LLM. I had also further hyperparameter tuning and defining the output components, outlining precisely what information and insights I aim to extract from the model's analyses.
Lastly I had develop the web application utilizing Streamlit.

<img width="366" alt="image" src="https://github.com/jasonsjafrudin/Tech-Virtual-Assistant-Large-Language-Model-LLM-/assets/61297201/e71da608-e608-425a-a17f-3901a2ccffb2">


